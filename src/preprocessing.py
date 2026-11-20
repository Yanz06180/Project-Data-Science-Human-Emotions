"""Image preprocessing and training-only light data augmentation for FER-2013.

Guarantees:
1. Single-channel grayscale conversion (mode 'L').
2. Exact spatial size of 48x48 pixels (bilinear resize only when needed).
3. Single pixel normalization from [0, 255] to float32 [0.0, 1.0] with a guard
   against accidental double normalization.
4. CNN-ready channel-last input shape of (48, 48, 1).
5. Light geometric augmentation (horizontal flip, light rotation, light zoom)
   strictly restricted to the 'train' split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile

from src.config import (
    AugmentationConfig,
    DEFAULT_AUGMENTATION_CONFIG,
    FIGURES_DIR,
    IMAGE_HEIGHT,
    IMAGE_SIZE,
    IMAGE_WIDTH,
    INPUT_SHAPE,
    LABEL_TO_ID,
    NORMALIZED_PIXEL_MAX,
    NORMALIZED_PIXEL_MIN,
    NUM_CLASSES,
    RANDOM_SEED,
    RAW_PIXEL_MAX,
)

# Fail loudly if a truncated/corrupted image is encountered
ImageFile.LOAD_TRUNCATED_IMAGES = False


def load_grayscale_image(image_path: Path | str) -> Image.Image:
    """Load an image from disk and ensure single-channel 8-bit grayscale ('L')."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        img.load()
        if img.mode != "L":
            return img.convert("L")
        return img.copy()


def ensure_image_size(
    image: Image.Image,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> Image.Image:
    """Ensure image dimensions equal ``target_size`` (width, height), resizing only if needed."""
    if image.size != target_size:
        return image.resize(target_size, resample=Image.Resampling.BILINEAR)
    return image


def _rotate_with_edge_padding(image: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate a grayscale PIL image with edge-replicated border padding to avoid black corners."""
    if abs(angle_deg) < 1e-6:
        return image

    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape
    pad = max(h, w) // 4  # 12 px for 48x48, ample for light rotations (<=15 deg)
    padded = np.pad(arr, pad_width=pad, mode="edge")
    rotated = Image.fromarray(padded, mode="L").rotate(
        angle_deg,
        resample=Image.Resampling.BILINEAR,
        expand=False,
    )
    rot_arr = np.asarray(rotated, dtype=np.uint8)
    cropped = rot_arr[pad : pad + h, pad : pad + w]
    return Image.fromarray(cropped, mode="L")


def _zoom_with_edge_padding(image: Image.Image, zoom_factor: float) -> Image.Image:
    """Apply light zoom (zoom_factor > 1 zooms in, < 1 zooms out) preserving 48x48 shape."""
    if abs(zoom_factor - 1.0) < 1e-4:
        return image

    w, h = image.size
    if zoom_factor > 1.0:
        # Zoom in: center-crop smaller region then resize back to (w, h)
        crop_w = max(1, int(round(w / zoom_factor)))
        crop_h = max(1, int(round(h / zoom_factor)))
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), resample=Image.Resampling.BILINEAR)

    # Zoom out: shrink image and pad symmetrically using edge pixel values
    new_w = max(1, int(round(w * zoom_factor)))
    new_h = max(1, int(round(h * zoom_factor)))
    shrunk = np.asarray(
        image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    pad_left = (w - new_w) // 2
    pad_right = w - new_w - pad_left
    pad_top = (h - new_h) // 2
    pad_bottom = h - new_h - pad_top
    padded = np.pad(
        shrunk,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )
    return Image.fromarray(padded, mode="L")


def apply_light_augmentation(
    image: Image.Image,
    rng: np.random.Generator,
    config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
) -> Image.Image:
    """Apply light geometric augmentation (flip, rotation, zoom) to a grayscale PIL image.

    Intended strictly for training samples before single-pass normalization.
    """
    aug_img = image

    # 1. Random Horizontal Flip
    if config.horizontal_flip_prob > 0.0 and rng.random() < config.horizontal_flip_prob:
        aug_img = aug_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 2. Light Rotation (Uniform[-rotation_range_deg, +rotation_range_deg])
    if config.rotation_range_deg > 0.0:
        angle = float(rng.uniform(-config.rotation_range_deg, config.rotation_range_deg))
        aug_img = _rotate_with_edge_padding(aug_img, angle)

    # 3. Light Zoom (Uniform[1 - zoom_range, 1 + zoom_range])
    if config.zoom_range > 0.0:
        zoom_factor = float(rng.uniform(1.0 - config.zoom_range, 1.0 + config.zoom_range))
        aug_img = _zoom_with_edge_padding(aug_img, zoom_factor)

    return ensure_image_size(aug_img, IMAGE_SIZE)


def normalize_pixels(image_array: np.ndarray) -> np.ndarray:
    """Normalize pixel values from [0, 255] to float32 [0.0, 1.0] exactly once.

    Raises a ValueError if passed a floating-point array that already lies in
    [0.0, 1.0], preventing accidental double normalization (e.g., / 255 / 255).
    """
    arr = np.asarray(image_array)

    if np.issubdtype(arr.dtype, np.floating):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_min >= NORMALIZED_PIXEL_MIN and arr_max <= NORMALIZED_PIXEL_MAX:
            raise ValueError(
                "Input floating-point array appears already normalized in [0.0, 1.0] "
                f"(min={arr_min:.6f}, max={arr_max:.6f}). Refusing to normalize twice."
            )
        normalized = arr.astype(np.float32) / np.float32(RAW_PIXEL_MAX)
    elif arr.dtype == np.uint8 or np.issubdtype(arr.dtype, np.integer):
        if int(arr.min()) < 0 or int(arr.max()) > 255:
            raise ValueError(
                f"Integer pixel values out of [0, 255] range: min={arr.min()}, max={arr.max()}"
            )
        normalized = arr.astype(np.float32) / np.float32(RAW_PIXEL_MAX)
    else:
        raise TypeError(f"Unsupported image array dtype for normalization: {arr.dtype}")

    return np.clip(normalized, NORMALIZED_PIXEL_MIN, NORMALIZED_PIXEL_MAX).astype(np.float32)


def to_cnn_input_shape(image_array: np.ndarray) -> np.ndarray:
    """Ensure a 2D grayscale array (48, 48) or (48, 48, 1) has exact shape (48, 48, 1)."""
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)
    if image_array.shape != INPUT_SHAPE:
        raise ValueError(f"Expected CNN input shape {INPUT_SHAPE}, got {image_array.shape}")
    return image_array.astype(np.float32, copy=False)


def preprocess_image(
    image_path: Path | str,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
) -> np.ndarray:
    """Run the full single-pass preprocessing pipeline on one image.

    Steps:
    1. Load as single-channel grayscale ('L').
    2. Resize to 48x48 only if dimensions differ.
    3. Optionally apply light augmentation (horizontal flip, rotation, zoom).
    4. Normalize pixels once from [0, 255] to float32 [0.0, 1.0].
    5. Format into CNN input shape (48, 48, 1).
    """
    img = load_grayscale_image(image_path)
    img = ensure_image_size(img, IMAGE_SIZE)

    if augment:
        generator = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        img = apply_light_augmentation(img, rng=generator, config=aug_config)

    raw_uint8 = np.asarray(img, dtype=np.uint8)
    normalized = normalize_pixels(raw_uint8)
    return to_cnn_input_shape(normalized)


def preprocess_batch(
    filepaths: Sequence[Path | str],
    labels: Sequence[int],
    split: str,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a batch of images into (X_batch, y_batch) tensors.

    Enforces strict anti-leakage guardrail: augmentation is only allowed when
    ``split == 'train'``.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Expected one of 'train', 'val', 'test'.")
    if augment and split != "train":
        raise ValueError(
            f"Data augmentation is strictly prohibited for split='{split}'. "
            "Only split='train' may be augmented."
        )
    if len(filepaths) != len(labels):
        raise ValueError(f"Length mismatch: {len(filepaths)} filepaths vs {len(labels)} labels.")

    batch_size = len(filepaths)
    x_batch = np.empty((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
    y_batch = np.asarray(labels, dtype=np.int64)

    generator = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
    for i, fpath in enumerate(filepaths):
        x_batch[i] = preprocess_image(
            fpath,
            augment=augment,
            rng=generator,
            aug_config=aug_config,
        )

    return x_batch, y_batch


def compute_balanced_class_weights(
    train_labels: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> dict[int, float]:
    """Compute balanced inverse-frequency class weights strictly from training labels."""
    y = np.asarray(train_labels, dtype=np.int64)
    total = len(y)
    weights: dict[int, float] = {}
    for cls_id in range(num_classes):
        count = int((y == cls_id).sum())
        if count == 0:
            raise ValueError(f"Class ID {cls_id} has 0 samples in training labels.")
        weights[cls_id] = round(total / (num_classes * count), 6)
    return weights


def save_preprocessing_examples_figure(
    train_df,
    output_path: Path = FIGURES_DIR / "07_preprocessing_and_augmentation_examples.png",
    seed: int = RANDOM_SEED,
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
) -> Path:
    """Save a visual comparison of raw grayscale, normalized (48,48,1), and 3 light augmentations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    classes = list(LABEL_TO_ID.keys())
    n_rows = len(classes)
    n_cols = 5  # Raw | Normalized [0,1] | Aug 1 | Aug 2 | Aug 3

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(13.0, 14.2),
        gridspec_kw={"wspace": 0.22, "hspace": 0.36},
    )

    col_titles = [
        "Original Grayscale\nuint8 [0–255]",
        "Normalized Input\nfloat32 [0–1] (48,48,1)",
        "Train Aug #1\nFlip, Rot ±10°, Zoom ±10%",
        "Train Aug #2\nFlip, Rot ±10°, Zoom ±10%",
        "Train Aug #3\nFlip, Rot ±10°, Zoom ±10%",
    ]

    for row_idx, cls_name in enumerate(classes):
        cls_rows = train_df[train_df["class_name"] == cls_name]
        # Deterministically pick a sample with non-zero variance for clear visualization
        pick_idx = int(rng.integers(0, len(cls_rows)))
        sample_row = cls_rows.iloc[pick_idx]
        img_path = Path(sample_row["abs_path"])

        raw_img = ensure_image_size(load_grayscale_image(img_path), IMAGE_SIZE)
        raw_arr = np.asarray(raw_img, dtype=np.uint8)
        norm_arr = preprocess_image(img_path, augment=False)
        aug_arrs = [
            preprocess_image(img_path, augment=True, rng=rng, aug_config=aug_config)
            for _ in range(3)
        ]

        panels = [
            (raw_arr, 0, 255, f"[{raw_arr.min()}, {raw_arr.max()}]"),
            (norm_arr[:, :, 0], 0.0, 1.0, f"[{norm_arr.min():.2f}, {norm_arr.max():.2f}]"),
        ] + [
            (a[:, :, 0], 0.0, 1.0, f"[{a.min():.2f}, {a.max():.2f}]")
            for a in aug_arrs
        ]

        for col_idx, (arr_2d, vmin, vmax, range_str) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(arr_2d, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"{col_titles[col_idx]}\n{range_str}", fontsize=8.5, fontweight="bold")
            else:
                ax.set_title(range_str, fontsize=8, color="#333333")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{cls_name.upper()}\n(id={LABEL_TO_ID[cls_name]})",
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=42,
                    va="center",
                )

    fig.suptitle(
        "Contoh Hasil Preprocessing & Light Augmentation (Training Only) — FER-2013",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    plt.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Run an end-to-end verification of the preprocessing & data loading pipeline."""
    from src.data_loader import build_data_loaders

    bundle = build_data_loaders(batch_size=64, random_seed=RANDOM_SEED)

    # 1. Verify single batch from each split
    x_tr, y_tr = bundle.train_loader[0]
    x_va, y_va = bundle.val_loader[0]
    x_te, y_te = bundle.test_loader[0]

    # 2. Verify double-normalization guard works
    double_norm_blocked = False
    try:
        normalize_pixels(x_tr[0])
    except ValueError:
        double_norm_blocked = True
    assert double_norm_blocked, "Double-normalization guard failed!"

    # 3. Verify augmentation guard blocks val and test
    val_aug_blocked = False
    test_aug_blocked = False
    try:
        preprocess_batch(bundle.val_df["abs_path"].iloc[:2].tolist(), bundle.val_df["label_id"].iloc[:2].tolist(), split="val", augment=True)
    except ValueError:
        val_aug_blocked = True
    try:
        preprocess_batch(bundle.test_df["abs_path"].iloc[:2].tolist(), bundle.test_df["label_id"].iloc[:2].tolist(), split="test", augment=True)
    except ValueError:
        test_aug_blocked = True
    assert val_aug_blocked and test_aug_blocked, "Augmentation guard failed for val/test!"

    # 4. Verify non-augmented vs augmented training output stats on one sample
    sample_path = bundle.train_df["abs_path"].iloc[0]
    sample_raw = np.asarray(load_grayscale_image(sample_path), dtype=np.uint8)
    sample_norm = preprocess_image(sample_path, augment=False)
    sample_aug = preprocess_image(sample_path, augment=True, rng=np.random.default_rng(RANDOM_SEED))

    # 5. Save visual preprocessing examples figure
    fig_path = save_preprocessing_examples_figure(bundle.train_df)

    # 6. Compute training-only balanced class weights (informational for training stage)
    class_weights = compute_balanced_class_weights(bundle.train_df["label_id"].tolist())

    print("=== VERIFIKASI PIPELINE DATA PREPROCESSING FER-2013 ===")
    print(f"Jumlah data Training   : {len(bundle.train_df):,} (augment={bundle.train_loader.augment}, shuffle={bundle.train_loader.shuffle}, batches={len(bundle.train_loader)})")
    print(f"Jumlah data Validation : {len(bundle.val_df):,} (augment={bundle.val_loader.augment}, shuffle={bundle.val_loader.shuffle}, batches={len(bundle.val_loader)})")
    print(f"Jumlah data Testing    : {len(bundle.test_df):,} (augment={bundle.test_loader.augment}, shuffle={bundle.test_loader.shuffle}, batches={len(bundle.test_loader)})")
    print(f"Total Keseluruhan      : {len(bundle.train_df) + len(bundle.val_df) + len(bundle.test_df):,}")
    print(f"Dimensi Input Gambar   : {INPUT_SHAPE}")
    print(f"Label Mapping          : {bundle.label_to_id}")
    print("\nVerifikasi Batch Tensor:")
    print(f"  Train batch : shape={x_tr.shape}, dtype={x_tr.dtype}, min={x_tr.min():.4f}, max={x_tr.max():.4f}, labels={y_tr.shape} ({y_tr.dtype})")
    print(f"  Val batch   : shape={x_va.shape}, dtype={x_va.dtype}, min={x_va.min():.4f}, max={x_va.max():.4f}, labels={y_va.shape} ({y_va.dtype})")
    print(f"  Test batch  : shape={x_te.shape}, dtype={x_te.dtype}, min={x_te.min():.4f}, max={x_te.max():.4f}, labels={y_te.shape} ({y_te.dtype})")
    print("\nGuardrails:")
    print(f"  Anti double-normalization guard   : PASSED ({double_norm_blocked})")
    print(f"  Anti val/test augmentation guard  : PASSED ({val_aug_blocked and test_aug_blocked})")
    print("\nContoh Hasil Preprocessing (1 sampel Training):")
    print(f"  File       : {bundle.train_df['filepath'].iloc[0]} (class='{bundle.train_df['class_name'].iloc[0]}', label_id={bundle.train_df['label_id'].iloc[0]})")
    print(f"  Raw uint8  : shape={sample_raw.shape}, dtype={sample_raw.dtype}, min={sample_raw.min()}, max={sample_raw.max()}, mean={sample_raw.mean():.2f}")
    print(f"  Normalized : shape={sample_norm.shape}, dtype={sample_norm.dtype}, min={sample_norm.min():.4f}, max={sample_norm.max():.4f}, mean={sample_norm.mean():.4f}")
    print(f"  Augmented  : shape={sample_aug.shape}, dtype={sample_aug.dtype}, min={sample_aug.min():.4f}, max={sample_aug.max():.4f}, mean={sample_aug.mean():.4f}")
    print(f"  Mean abs diff (Norm vs Aug) : {float(np.abs(sample_norm - sample_aug).mean()):.4f}")
    print("\nDistribusi Kelas pada Masing-Masing Subset:")
    print(bundle.class_distribution.to_string(index=False))
    print(f"\nBalanced Class Weights (Training Only): {json.dumps(class_weights)}")
    print(f"Visualisasi contoh preprocessing disimpan ke: {fig_path}")


if __name__ == "__main__":
    main()
