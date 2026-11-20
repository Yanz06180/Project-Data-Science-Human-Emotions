"""Modular Stratified Data Splitting for FER-2013 (Train / Validation).

Splits images from ``data/raw/train/`` into reproducible 80% Training and
20% Validation subsets using deterministic stratified splitting, while keeping
``data/raw/test/`` completely isolated and untouched.

Original files under ``data/raw/`` are never modified, copied, or moved.
Outputs are stored as lightweight CSV manifests and JSON metadata in
``data/splits/`` so preprocessing and training pipelines can load splits
consistently via :func:`load_split`, :func:`load_train_val_splits`, and
:func:`load_label_mapping`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    RANDOM_SEED as DEFAULT_RANDOM_SEED,
    RAW_DATA_DIR as DEFAULT_RAW_DIR,
    SPLITS_DIR as DEFAULT_SPLITS_DIR,
    VALID_IMAGE_EXTENSIONS,
    VAL_RATIO as DEFAULT_VAL_RATIO,
)


def build_label_mapping(train_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic alphabetical class-to-ID and ID-to-class mappings."""
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class subdirectories found in {train_dir}")

    label_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}
    id_to_label = {idx: cls_name for idx, cls_name in enumerate(classes)}
    return label_to_id, id_to_label


def collect_directory_manifest(
    split_dir: Path,
    label_to_id: dict[str, int],
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Scan a raw split directory read-only and build a sorted file manifest."""
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    found_classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    expected_classes = sorted(label_to_id.keys())
    if found_classes != expected_classes:
        raise ValueError(
            f"Category mismatch in {split_dir}: found {found_classes}, expected {expected_classes}"
        )

    rows: list[dict[str, Any]] = []
    for cls_name in expected_classes:
        cls_dir = split_dir / cls_name
        label_id = label_to_id[cls_name]
        # Deterministic sort by filename
        files = sorted(
            (p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS),
            key=lambda p: p.name,
        )
        for fpath in files:
            rel_posix = fpath.resolve().relative_to(project_root.resolve()).as_posix()
            rows.append({
                "filepath": rel_posix,
                "filename": fpath.name,
                "class_name": cls_name,
                "label_id": label_id,
            })

    df = pd.DataFrame(rows, columns=["filepath", "filename", "class_name", "label_id"])
    return df


def _allocate_stratified_counts(
    class_counts: dict[str, int],
    val_ratio: float,
) -> dict[str, int]:
    """Compute per-class validation counts using the Largest Remainder Method.

    Ensures the total validation count equals ``round(total * val_ratio)``
    while keeping every single class within 1 sample of its exact proportion
    and guaranteeing at least 1 sample in both train and val per class.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be strictly between 0 and 1, got {val_ratio}")

    total = sum(class_counts.values())
    target_val_total = int(round(total * val_ratio))

    exact_quotas = {cls: count * val_ratio for cls, count in class_counts.items()}
    base_alloc = {cls: int(math.floor(q)) for cls, q in exact_quotas.items()}
    remainders = {cls: exact_quotas[cls] - base_alloc[cls] for cls in class_counts}

    # Ensure every class has >= 1 in val and >= 1 in train
    for cls, count in class_counts.items():
        if count < 2:
            raise ValueError(f"Class '{cls}' has only {count} sample(s); cannot stratify.")
        base_alloc[cls] = max(1, min(count - 1, base_alloc[cls]))

    remainder_needed = target_val_total - sum(base_alloc.values())
    if remainder_needed > 0:
        # Sort by largest remainder descending, tie-break deterministically by class name
        for cls in sorted(class_counts.keys(), key=lambda c: (-remainders[c], c)):
            if remainder_needed == 0:
                break
            if base_alloc[cls] < class_counts[cls] - 1:
                base_alloc[cls] += 1
                remainder_needed -= 1
    elif remainder_needed < 0:
        for cls in sorted(class_counts.keys(), key=lambda c: (remainders[c], c)):
            if remainder_needed == 0:
                break
            if base_alloc[cls] > 1:
                base_alloc[cls] -= 1
                remainder_needed += 1

    return base_alloc


def stratified_train_val_split(
    raw_train_df: pd.DataFrame,
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform reproducible stratified split of raw training data into train and val.

    Input rows are sorted deterministically by ``(label_id, filepath)`` prior to
    per-class permutation with ``numpy.random.default_rng(random_seed)``.
    """
    required_cols = {"filepath", "filename", "class_name", "label_id"}
    missing = required_cols - set(raw_train_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw_train_df: {sorted(missing)}")

    ordered_df = raw_train_df.sort_values(["label_id", "filepath"]).reset_index(drop=True)
    class_order = (
        ordered_df[["class_name", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")["class_name"]
        .tolist()
    )
    class_counts = {
        cls: int((ordered_df["class_name"] == cls).sum())
        for cls in class_order
    }
    val_counts = _allocate_stratified_counts(class_counts, val_ratio)

    rng = np.random.default_rng(random_seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for cls in class_order:
        cls_df = ordered_df[ordered_df["class_name"] == cls].reset_index(drop=True)
        n_cls = len(cls_df)
        n_val = val_counts[cls]

        perm = rng.permutation(n_cls)
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])

        train_parts.append(cls_df.iloc[train_idx])
        val_parts.append(cls_df.iloc[val_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    train_df["split"] = "train"
    val_df["split"] = "val"

    # Sort deterministically for clean, diffable manifests
    train_df = train_df.sort_values(["label_id", "filepath"]).reset_index(drop=True)
    val_df = val_df.sort_values(["label_id", "filepath"]).reset_index(drop=True)
    return train_df, val_df


def _check_content_hash_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, int]:
    """Read-only diagnostic checking if identical raw image bytes occur across train & val."""
    train_hashes: dict[str, int] = {}
    for rel_path in train_df["filepath"]:
        digest = hashlib.sha256((project_root / rel_path).read_bytes()).hexdigest()
        train_hashes[digest] = train_hashes.get(digest, 0) + 1

    val_files_with_same_bytes_in_train = 0
    shared_unique_hashes: set[str] = set()
    for rel_path in val_df["filepath"]:
        digest = hashlib.sha256((project_root / rel_path).read_bytes()).hexdigest()
        if digest in train_hashes:
            val_files_with_same_bytes_in_train += 1
            shared_unique_hashes.add(digest)

    return {
        "shared_unique_content_hashes_train_val": len(shared_unique_hashes),
        "val_files_matching_train_content_bytes": val_files_with_same_bytes_in_train,
    }


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_to_id: dict[str, int],
    expected_raw_train_total: int,
    random_seed: int,
    val_ratio: float,
    raw_train_df: pd.DataFrame | None = None,
    check_content_overlap: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify integrity, disjointness, stratification, label consistency, and reproducibility."""
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    # 1. Zero duplicate paths within subsets and zero overlap across subsets
    assert len(train_paths) == len(train_df), "Duplicate filepaths detected inside train split!"
    assert len(val_paths) == len(val_df), "Duplicate filepaths detected inside val split!"
    assert len(test_paths) == len(test_df), "Duplicate filepaths detected inside test manifest!"
    assert train_paths.isdisjoint(val_paths), "Overlap detected between train and val splits!"
    assert test_paths.isdisjoint(train_paths), "Test files leaked into train split!"
    assert test_paths.isdisjoint(val_paths), "Test files leaked into val split!"

    # 2. Exact conservation of raw training samples
    assert len(train_df) + len(val_df) == expected_raw_train_total, (
        f"Sample count mismatch: {len(train_df)} + {len(val_df)} != {expected_raw_train_total}"
    )

    # 3. Verify no test path prefix exists in train/val and all train/val paths come from data/raw/train/
    assert all(p.startswith("data/raw/train/") for p in train_paths), "Unexpected non-train source in train split!"
    assert all(p.startswith("data/raw/train/") for p in val_paths), "Unexpected non-train source in val split!"
    assert all(p.startswith("data/raw/test/") for p in test_paths), "Unexpected non-test source in test manifest!"

    # 4. All classes represented in train, val, and test
    expected_classes = set(label_to_id.keys())
    assert set(train_df["class_name"]) == expected_classes, "Missing classes in train split!"
    assert set(val_df["class_name"]) == expected_classes, "Missing classes in val split!"
    assert set(test_df["class_name"]) == expected_classes, "Missing classes in test manifest!"

    # 5. Consistent label mapping across all subsets
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        mapped = df["class_name"].map(label_to_id)
        assert (mapped == df["label_id"]).all(), f"Label mapping inconsistency detected in {name}!"

    # 6. Stratification drift check
    train_pcts = train_df["class_name"].value_counts(normalize=True) * 100
    val_pcts = val_df["class_name"].value_counts(normalize=True) * 100
    max_prop_diff_pct = float(max(abs(train_pcts[c] - val_pcts[c]) for c in expected_classes))
    assert max_prop_diff_pct < 0.1, f"Stratification proportion drift too large: {max_prop_diff_pct:.4f}%"

    # 7. Reproducibility check
    reproducible = True
    if raw_train_df is not None:
        train_df_2, val_df_2 = stratified_train_val_split(
            raw_train_df, val_ratio=val_ratio, random_seed=random_seed
        )
        reproducible = bool(train_df.equals(train_df_2) and val_df.equals(val_df_2))
        assert reproducible, "Splitting is not reproducible with the same random seed!"

    checks: dict[str, Any] = {
        "no_train_val_filepath_overlap": True,
        "no_test_leakage_into_train_or_val": True,
        "exact_raw_train_conservation": True,
        "all_classes_present_in_train_and_val": True,
        "label_mapping_consistent_across_all_splits": True,
        "max_class_proportion_diff_pct": round(max_prop_diff_pct, 4),
        "reproducible_with_same_seed": reproducible,
    }

    if check_content_overlap:
        checks["raw_content_duplicate_diagnostic"] = _check_content_hash_overlap(
            train_df, val_df, project_root=project_root
        )

    return checks


def build_summary_table(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_to_id: dict[str, int],
) -> pd.DataFrame:
    """Build per-class summary table for train, val, and untouched test."""
    rows = []
    train_total = len(train_df)
    val_total = len(val_df)
    test_total = len(test_df)

    for cls_name, label_id in sorted(label_to_id.items(), key=lambda kv: kv[1]):
        n_tr = int((train_df["class_name"] == cls_name).sum())
        n_va = int((val_df["class_name"] == cls_name).sum())
        n_raw_tr = n_tr + n_va
        n_te = int((test_df["class_name"] == cls_name).sum())
        rows.append({
            "label_id": label_id,
            "class_name": cls_name,
            "raw_train_total": n_raw_tr,
            "train_count": n_tr,
            "train_pct": round(n_tr / train_total * 100, 2),
            "val_count": n_va,
            "val_pct": round(n_va / val_total * 100, 2),
            "class_val_ratio_pct": round(n_va / n_raw_tr * 100, 2),
            "test_count_untouched": n_te,
            "test_pct": round(n_te / test_total * 100, 2),
        })

    return pd.DataFrame(rows)


def create_and_save_splits(
    raw_dir: Path = DEFAULT_RAW_DIR,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    check_content_overlap: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute end-to-end stratified train/val splitting and persist modular artifacts."""
    raw_train_dir = raw_dir / "train"
    raw_test_dir = raw_dir / "test"

    label_to_id, id_to_label = build_label_mapping(raw_train_dir)
    raw_train_df = collect_directory_manifest(raw_train_dir, label_to_id, project_root=project_root)

    # Test is collected strictly as a read-only manifest; NEVER passed into splitting
    test_df = collect_directory_manifest(raw_test_dir, label_to_id, project_root=project_root)
    test_df["split"] = "test"

    train_df, val_df = stratified_train_val_split(
        raw_train_df,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    verification = validate_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_to_id=label_to_id,
        expected_raw_train_total=len(raw_train_df),
        random_seed=random_seed,
        val_ratio=val_ratio,
        raw_train_df=raw_train_df,
        check_content_overlap=check_content_overlap,
        project_root=project_root,
    )

    summary_df = build_summary_table(train_df, val_df, test_df, label_to_id)

    # Persist lightweight manifests & metadata (data/raw/ remains 100% untouched)
    splits_dir.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "filename", "class_name", "label_id", "split"]
    train_df[cols].to_csv(splits_dir / "train.csv", index=False)
    val_df[cols].to_csv(splits_dir / "val.csv", index=False)
    test_df[cols].to_csv(splits_dir / "test.csv", index=False)

    label_mapping_payload = {
        "num_classes": len(label_to_id),
        "classes": list(label_to_id.keys()),
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
    }
    (splits_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping_payload, indent=2), encoding="utf-8"
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_train_dir": raw_train_dir.resolve().relative_to(project_root.resolve()).as_posix(),
        "source_test_dir_untouched": raw_test_dir.resolve().relative_to(project_root.resolve()).as_posix(),
        "random_seed": random_seed,
        "train_ratio": round(1.0 - val_ratio, 4),
        "val_ratio": round(val_ratio, 4),
        "counts": {
            "raw_train_total": int(len(raw_train_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test_separate": int(len(test_df)),
            "grand_total": int(len(raw_train_df) + len(test_df)),
        },
        "actual_ratios": {
            "train_pct_of_raw_train": round(len(train_df) / len(raw_train_df) * 100, 4),
            "val_pct_of_raw_train": round(len(val_df) / len(raw_train_df) * 100, 4),
        },
        "label_mapping": label_to_id,
        "verification": verification,
        "per_class_summary": summary_df.to_dict(orient="records"),
    }
    (splits_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


# ---------------------------------------------------------------------------
# Reusable loader API for downstream preprocessing & training stages
# ---------------------------------------------------------------------------

def load_label_mapping(
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> tuple[dict[str, int], dict[int, str]]:
    """Load the persisted consistent label mapping."""
    mapping_path = splits_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Label mapping not found at {mapping_path}. Run src/data_splitting.py first."
        )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    label_to_id = {str(k): int(v) for k, v in payload["label_to_id"].items()}
    id_to_label = {int(k): str(v) for k, v in payload["id_to_label"].items()}
    return label_to_id, id_to_label


def load_split(
    split_name: str,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    resolve_paths: bool = True,
) -> pd.DataFrame:
    """Load a persisted split manifest ('train', 'val', or 'test') safely.

    Validates label mapping consistency against ``label_mapping.json`` and
    optionally adds an ``abs_path`` column for direct image loading.
    """
    valid_names = {"train", "val", "test"}
    if split_name not in valid_names:
        raise ValueError(f"split_name must be one of {sorted(valid_names)}, got '{split_name}'")

    csv_path = splits_dir / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {csv_path}. Run src/data_splitting.py first."
        )

    label_to_id, _ = load_label_mapping(splits_dir=splits_dir)
    df = pd.read_csv(csv_path)

    # Sanity check consistent mapping on load
    if not (df["class_name"].map(label_to_id) == df["label_id"]).all():
        raise ValueError(f"Inconsistent label_id mapping detected in {csv_path}")

    if resolve_paths:
        df["abs_path"] = df["filepath"].map(lambda rel: str((project_root / rel).resolve()))

    return df


def load_train_val_splits(
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    project_root: Path = PROJECT_ROOT,
    resolve_paths: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience loader returning ``(train_df, val_df)`` without touching test."""
    train_df = load_split("train", splits_dir=splits_dir, project_root=project_root, resolve_paths=resolve_paths)
    val_df = load_split("val", splits_dir=splits_dir, project_root=project_root, resolve_paths=resolve_paths)
    return train_df, val_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified 80/20 Train/Val Splitter for FER-2013.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation ratio (default: 0.20)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Path to data/raw directory")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Output directory for split manifests")
    parser.add_argument("--skip-content-check", action="store_true", help="Skip raw-byte content hash diagnostic")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = create_and_save_splits(
        raw_dir=args.raw_dir,
        splits_dir=args.splits_dir,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        check_content_overlap=not args.skip_content_check,
    )
    print("=== STRATIFIED DATA SPLITTING COMPLETE ===")
    print(f"Random Seed           : {metadata['random_seed']}")
    print(f"Split Ratio           : {metadata['train_ratio']*100:.0f}% Train / {metadata['val_ratio']*100:.0f}% Validation")
    print(f"Raw TRAIN Total       : {metadata['counts']['raw_train_total']:,}")
    print(f"  -> Training Subset  : {metadata['counts']['train']:,} ({metadata['actual_ratios']['train_pct_of_raw_train']:.2f}%)")
    print(f"  -> Validation Subset: {metadata['counts']['val']:,} ({metadata['actual_ratios']['val_pct_of_raw_train']:.2f}%)")
    print(f"Untouched TEST Subset : {metadata['counts']['test_separate']:,} (strictly excluded from splitting)")
    print(f"Label Mapping         : {metadata['label_mapping']}")
    print(f"Verification Checks   : {json.dumps(metadata['verification'], indent=2)}")
    print("\nPer-Class Distribution:")
    print(pd.DataFrame(metadata["per_class_summary"]).to_string(index=False))
    print(f"\nSaved manifests to: {args.splits_dir}")


if __name__ == "__main__":
    main()
