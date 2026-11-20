"""Modular dataset loading and batch iteration for FER-2013 CNN pipelines.

Reuses the persisted stratified train/val splits and untouched test manifest in
``data/splits/`` (created by ``src/data_splitting.py``). If splits have not yet
been generated, triggers stratified 80/20 splitting on ``data/raw/train/`` only
without ever including ``data/raw/test/``.

Applies light data augmentation strictly to the ``train`` loader and forbids
augmentation on ``val`` and ``test`` loaders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import (
    AugmentationConfig,
    DEFAULT_AUGMENTATION_CONFIG,
    DEFAULT_BATCH_SIZE,
    INPUT_SHAPE,
    LABEL_TO_ID,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SPLITS_DIR,
    VAL_RATIO,
)
from src.data_splitting import (
    create_and_save_splits,
    load_label_mapping,
    load_split,
    load_train_val_splits,
)
from src.preprocessing import preprocess_batch


def ensure_splits_available(
    splits_dir: Path = SPLITS_DIR,
    raw_dir: Path = RAW_DATA_DIR,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> bool:
    """Reuse existing persisted splits if complete; otherwise generate them once.

    Returns ``True`` if existing splits were reused, ``False`` if newly created.
    """
    required_files = [
        splits_dir / "train.csv",
        splits_dir / "val.csv",
        splits_dir / "test.csv",
        splits_dir / "label_mapping.json",
    ]
    if all(p.exists() for p in required_files):
        return True

    create_and_save_splits(
        raw_dir=raw_dir,
        splits_dir=splits_dir,
        val_ratio=val_ratio,
        random_seed=random_seed,
        check_content_overlap=False,
    )
    return False


def verify_label_mapping_matches_config(
    label_to_id: dict[str, int],
) -> None:
    """Assert that persisted label mapping matches canonical ``src.config.LABEL_TO_ID``."""
    if label_to_id != LABEL_TO_ID:
        raise ValueError(
            f"Label mapping mismatch! Persisted={label_to_id}, Config={LABEL_TO_ID}"
        )


def compute_subset_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_to_id: dict[str, int] = LABEL_TO_ID,
) -> pd.DataFrame:
    """Build a per-class count & percentage distribution table across train, val, and test."""
    rows = []
    n_tr_total = len(train_df)
    n_va_total = len(val_df)
    n_te_total = len(test_df)

    for cls_name, label_id in sorted(label_to_id.items(), key=lambda kv: kv[1]):
        n_tr = int((train_df["label_id"] == label_id).sum())
        n_va = int((val_df["label_id"] == label_id).sum())
        n_te = int((test_df["label_id"] == label_id).sum())
        rows.append({
            "label_id": label_id,
            "class_name": cls_name,
            "train_count": n_tr,
            "train_pct": round(n_tr / n_tr_total * 100, 2),
            "val_count": n_va,
            "val_pct": round(n_va / n_va_total * 100, 2),
            "test_count": n_te,
            "test_pct": round(n_te / n_te_total * 100, 2),
            "total": n_tr + n_va + n_te,
        })

    return pd.DataFrame(rows)


class FER2013DataLoader:
    """Framework-agnostic, reproducible batch loader yielding ``(X_batch, y_batch)``.

    - Output ``X_batch``: ``float32`` tensor of shape ``(B, 48, 48, 1)`` in ``[0.0, 1.0]``.
    - Output ``y_batch``: ``int64`` tensor of shape ``(B,)`` with IDs ``0..6``.
    - Augmentation is strictly permitted ONLY for ``split == 'train'``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle: bool | None = None,
        augment: bool | None = None,
        random_seed: int = RANDOM_SEED,
        aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Default policy: shuffle & augment ONLY on training
        resolved_shuffle = (split == "train") if shuffle is None else shuffle
        resolved_augment = (split == "train") if augment is None else augment

        if resolved_augment and split != "train":
            raise ValueError(
                f"Augmentation is forbidden for split='{split}'. "
                "Validation and testing sets must never be augmented."
            )

        self.df = df.reset_index(drop=True)
        self.split = split
        self.batch_size = int(batch_size)
        self.shuffle = bool(resolved_shuffle)
        self.augment = bool(resolved_augment)
        self.random_seed = int(random_seed)
        self.aug_config = aug_config

        self._filepaths = self.df["abs_path"].tolist()
        self._labels = self.df["label_id"].astype(np.int64).to_numpy()
        self._indices = np.arange(len(self.df), dtype=np.int64)
        self._epoch = 0

        if self.shuffle:
            self._shuffle_indices(epoch=0)

    def _shuffle_indices(self, epoch: int) -> None:
        rng = np.random.default_rng(self.random_seed + epoch)
        self._indices = rng.permutation(len(self.df))

    def on_epoch_end(self) -> None:
        """Advance epoch counter and deterministically re-shuffle if shuffle=True."""
        self._epoch += 1
        if self.shuffle:
            self._shuffle_indices(epoch=self._epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError(f"Batch index {batch_idx} out of range [0, {len(self) - 1}]")

        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.df))
        batch_indices = self._indices[start:end]

        batch_paths = [self._filepaths[i] for i in batch_indices]
        batch_labels = self._labels[batch_indices]

        # Deterministic per-epoch, per-batch RNG for reproducible augmentation
        batch_rng = np.random.default_rng(self.random_seed + self._epoch * 100_003 + batch_idx)
        return preprocess_batch(
            filepaths=batch_paths,
            labels=batch_labels,
            split=self.split,
            augment=self.augment,
            rng=batch_rng,
            aug_config=self.aug_config,
        )

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for idx in range(len(self)):
            yield self[idx]

    def load_all_arrays(self, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Load the full split into memory as ``(X, y)`` with shape ``(N, 48, 48, 1)``."""
        if augment and self.split != "train":
            raise ValueError(f"Augmentation is forbidden for split='{self.split}'.")
        rng = np.random.default_rng(self.random_seed)
        return preprocess_batch(
            filepaths=self._filepaths,
            labels=self._labels,
            split=self.split,
            augment=augment,
            rng=rng,
            aug_config=self.aug_config,
        )


@dataclass(frozen=True)
class TrainValDataBundle:
    """Training-only container holding strictly train and validation loaders (NO test data)."""

    train_loader: FER2013DataLoader
    val_loader: FER2013DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    input_shape: tuple[int, int, int]
    reused_existing_splits: bool


def build_train_val_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
    splits_dir: Path = SPLITS_DIR,
    raw_dir: Path = RAW_DATA_DIR,
    project_root: Path = PROJECT_ROOT,
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
    train_augment: bool = False,
) -> TrainValDataBundle:
    """Build ONLY training and validation loaders without ever accessing test data.

    Designed for ``src/train.py`` so ``data/raw/test/`` and ``data/splits/test.csv``
    remain strictly untouched during model training.
    """
    reused = ensure_splits_available(
        splits_dir=splits_dir,
        raw_dir=raw_dir,
        val_ratio=VAL_RATIO,
        random_seed=random_seed,
    )

    label_to_id, id_to_label = load_label_mapping(splits_dir=splits_dir)
    verify_label_mapping_matches_config(label_to_id)

    train_df, val_df = load_train_val_splits(
        splits_dir=splits_dir,
        project_root=project_root,
        resolve_paths=True,
    )

    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    assert train_paths.isdisjoint(val_paths), "Overlap detected between train and val!"
    assert all(p.startswith("data/raw/train/") for p in train_paths | val_paths), (
        "Non-training source detected in train/val splits!"
    )

    train_loader = FER2013DataLoader(
        df=train_df,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        augment=train_augment,
        random_seed=random_seed,
        aug_config=aug_config,
    )
    val_loader = FER2013DataLoader(
        df=val_df,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        random_seed=random_seed,
        aug_config=aug_config,
    )

    return TrainValDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_df,
        val_df=val_df,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        input_shape=INPUT_SHAPE,
        reused_existing_splits=reused,
    )


@dataclass(frozen=True)
class DataPipelineBundle:
    """Container holding all verified split manifests, loaders, and metadata."""

    train_loader: FER2013DataLoader
    val_loader: FER2013DataLoader
    test_loader: FER2013DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    class_distribution: pd.DataFrame
    input_shape: tuple[int, int, int]
    reused_existing_splits: bool


def build_data_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
    splits_dir: Path = SPLITS_DIR,
    raw_dir: Path = RAW_DATA_DIR,
    project_root: Path = PROJECT_ROOT,
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
    train_augment: bool = True,
) -> DataPipelineBundle:
    """Build verified train, validation, and testing data loaders cleanly.

    - Reuses persisted splits from ``data/splits/`` without creating a second split.
    - Guarantees zero path overlap between train, val, and test.
    - Applies light augmentation ONLY to ``train_loader`` when ``train_augment=True``
      (set ``train_augment=False`` if using Keras in-model augmentation to avoid
      applying augmentation twice).
    """
    reused = ensure_splits_available(
        splits_dir=splits_dir,
        raw_dir=raw_dir,
        val_ratio=VAL_RATIO,
        random_seed=random_seed,
    )

    label_to_id, id_to_label = load_label_mapping(splits_dir=splits_dir)
    verify_label_mapping_matches_config(label_to_id)

    train_df, val_df = load_train_val_splits(
        splits_dir=splits_dir,
        project_root=project_root,
        resolve_paths=True,
    )
    test_df = load_split(
        "test",
        splits_dir=splits_dir,
        project_root=project_root,
        resolve_paths=True,
    )

    # Disjointness & source isolation guardrails
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])
    assert train_paths.isdisjoint(val_paths), "Overlap detected between train and val!"
    assert test_paths.isdisjoint(train_paths | val_paths), "Test leakage detected into train/val!"
    assert all(p.startswith("data/raw/train/") for p in train_paths | val_paths)
    assert all(p.startswith("data/raw/test/") for p in test_paths)

    class_distribution = compute_subset_class_distribution(train_df, val_df, test_df, label_to_id)

    train_loader = FER2013DataLoader(
        df=train_df,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        augment=train_augment,
        random_seed=random_seed,
        aug_config=aug_config,
    )
    val_loader = FER2013DataLoader(
        df=val_df,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        random_seed=random_seed,
        aug_config=aug_config,
    )
    test_loader = FER2013DataLoader(
        df=test_df,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        random_seed=random_seed,
        aug_config=aug_config,
    )

    return DataPipelineBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        class_distribution=class_distribution,
        input_shape=INPUT_SHAPE,
        reused_existing_splits=reused,
    )


def to_tf_dataset(loader: FER2013DataLoader):
    """Optional adapter wrapping ``FER2013DataLoader`` into a ``tf.data.Dataset``."""
    import tensorflow as tf

    def _gen():
        for x_batch, y_batch in loader:
            yield x_batch, y_batch

    output_signature = (
        tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    )
    return tf.data.Dataset.from_generator(_gen, output_signature=output_signature).prefetch(
        tf.data.AUTOTUNE
    )


def main() -> None:
    bundle = build_data_loaders()
    print("=== FER-2013 DATA LOADER SUMMARY ===")
    print(f"Reused existing splits : {bundle.reused_existing_splits}")
    print(f"Input shape            : {bundle.input_shape}")
    print(f"Label mapping          : {bundle.label_to_id}")
    print(f"Train samples / batches: {len(bundle.train_df):,} / {len(bundle.train_loader)} (augment={bundle.train_loader.augment})")
    print(f"Val samples   / batches: {len(bundle.val_df):,} / {len(bundle.val_loader)} (augment={bundle.val_loader.augment})")
    print(f"Test samples  / batches: {len(bundle.test_df):,} / {len(bundle.test_loader)} (augment={bundle.test_loader.augment})")
    print("\nDistribusi Kelas:")
    print(bundle.class_distribution.to_string(index=False))


if __name__ == "__main__":
    main()
