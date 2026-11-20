"""Model Training Pipeline for FER-2013 Deep Learning Models (TensorFlow & Keras).

Mendukung eksperimen modular melalui model factory/registry (``baseline``,
``improved``, ``mlp``, dan slot ekspansi ``mobilenetv2``) dengan isolasi artefak
penuh agar hasil ``baseline`` dan ``exp2`` tidak pernah tertimpa.

Jaminan Metodologis:
1. HANYA subset Training (22.967 citra) dan Validation (5.742 citra) yang dimuat.
   Subset Testing (``data/raw/test/`` / ``data/splits/test.csv``) TIDAK pernah
   diakses ataupun digunakan selama proses training maupun perbandingan model.
2. Normalisasi piksel [0, 255] -> [0.0, 1.0] dilakukan tepat SATU kali oleh
   ``src/preprocessing.py`` (dilindungi anti-double-normalization guard).
3. Data augmentation ringan aktif HANYA di dalam model saat ``training=True``
   (``train_augment=False`` pada loader agar tidak terjadi augmentasi ganda;
   validation selalu dievaluasi tanpa augmentasi).
4. Mencatat metrik deployment PRD (``model_size_mb`` dan
   ``val_inference_time_ms_per_image`` dengan 1-batch warm-up) serta mendukung
   resumable training (``--resume``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    EXP2_BEST_MODEL_PATH,
    EXP2_LAST_MODEL_PATH,
    FIGURES_DIR,
    ID_TO_LABEL,
    INPUT_SHAPE,
    LABEL_TO_ID,
    LAST_MODEL_PATH,
    MAX_EPOCHS,
    MODELS_DIR,
    NUM_CLASSES,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_LR,
    REDUCE_LR_PATIENCE,
    REPORTS_DIR,
    SPLITS_DIR,
    TRAINING_HISTORY_CSV,
    TRAINING_HISTORY_JSON,
    TRAINING_REPORT_MD,
    TRAIN_VAL_CACHE_PATH,
)
from src.data_loader import build_train_val_loaders
from src.metrics import CLASS_DISPLAY_NAMES, compute_classification_metrics
from src.model import build_improved_model, build_mlp_model, build_model, compile_model
from src.preprocessing import compute_balanced_class_weights, normalize_pixels


def _build_mobilenetv2_placeholder(*args: Any, **kwargs: Any) -> keras.Model:
    raise NotImplementedError(
        "Model variant 'mobilenetv2' belum diimplementasikan. "
        "Tambahkan build_mobilenetv2_model() di src/model.py lalu daftarkan pada MODEL_REGISTRY."
    )


MODEL_REGISTRY: dict[str, Callable[..., keras.Model]] = {
    "baseline": build_model,
    "improved": build_improved_model,
    "mlp": build_mlp_model,
    "mobilenetv2": _build_mobilenetv2_placeholder,
}

VARIANT_DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Baseline CNN",
    "improved": "Improved CNN",
    "mlp": "MLP",
    "mobilenetv2": "MobileNetV2",
}


def configure_gpu() -> str:
    """Konfigurasi GPU memory growth secara aman; tetap berjalan normal pada CPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return f"GPU ({len(gpus)} device)"
    return "CPU (TensorFlow oneDNN)"


def create_model_by_variant(
    model_variant: str,
    seed: int = RANDOM_SEED,
    use_data_augmentation: bool = True,
) -> keras.Model:
    """Factory builder untuk memilih arsitektur model secara terpusat."""
    if model_variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. Supported: {sorted(MODEL_REGISTRY.keys())}"
        )
    builder = MODEL_REGISTRY[model_variant]
    return builder(use_data_augmentation=use_data_augmentation, seed=seed)


class EpochProgressCallback(keras.callbacks.Callback):
    """Mencatat durasi wall-clock dan menampilkan progress ringkas setiap epoch."""

    def __init__(self, total_epochs: int) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_times: list[float] = []
        self._epoch_start: float = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append(round(elapsed, 3))
        logs = logs or {}

        lr = float(self.model.optimizer.learning_rate.numpy())
        loss = float(logs.get("loss", float("nan")))
        acc = float(logs.get("accuracy", float("nan")))
        val_loss = float(logs.get("val_loss", float("nan")))
        val_acc = float(logs.get("val_accuracy", float("nan")))

        print(
            f"Epoch {epoch + 1:02d}/{self.total_epochs:02d} "
            f"[{elapsed:5.1f}s] | lr={lr:.2e} | "
            f"loss={loss:.4f} | acc={acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}",
            flush=True,
        )


def resolve_experiment_paths(experiment: str) -> dict[str, Path]:
    """Menentukan path artefak terisolasi per eksperimen (backward-compatible untuk baseline & exp2)."""
    if experiment == "baseline":
        return {
            "best_model": BEST_MODEL_PATH,
            "last_model": LAST_MODEL_PATH,
            "history_csv": TRAINING_HISTORY_CSV,
            "history_json": TRAINING_HISTORY_JSON,
            "report_md": TRAINING_REPORT_MD,
            "acc_figure": FIGURES_DIR / "08_train_val_accuracy.png",
            "loss_figure": FIGURES_DIR / "09_train_val_loss.png",
            "val_compare_figure": FIGURES_DIR / "08_val_comparison.png",
        }
    if experiment == "exp2":
        return {
            "best_model": EXP2_BEST_MODEL_PATH,
            "last_model": EXP2_LAST_MODEL_PATH,
            "history_csv": REPORTS_DIR / "exp2_training_history.csv",
            "history_json": REPORTS_DIR / "exp2_training_history.json",
            "report_md": REPORTS_DIR / "exp2_training_report.md",
            "acc_figure": FIGURES_DIR / "13_exp2_train_val_accuracy.png",
            "loss_figure": FIGURES_DIR / "14_exp2_train_val_loss.png",
            "val_compare_figure": FIGURES_DIR / "15_val_baseline_vs_exp2.png",
        }
    return {
        "best_model": MODELS_DIR / f"fer2013_{experiment}.keras",
        "last_model": MODELS_DIR / f"fer2013_{experiment}_last.keras",
        "history_csv": REPORTS_DIR / f"{experiment}_training_history.csv",
        "history_json": REPORTS_DIR / f"{experiment}_training_history.json",
        "report_md": REPORTS_DIR / f"{experiment}_training_report.md",
        "acc_figure": FIGURES_DIR / f"{experiment}_train_val_accuracy.png",
        "loss_figure": FIGURES_DIR / f"{experiment}_train_val_loss.png",
        "val_compare_figure": FIGURES_DIR / f"{experiment}_val_vs_baseline.png",
    }


def _splits_signature(splits_dir: Path = SPLITS_DIR) -> str:
    h = hashlib.sha256()
    for name in ("train.csv", "val.csv", "label_mapping.json"):
        h.update((splits_dir / name).read_bytes())
    return h.hexdigest()


def load_train_val_arrays_cached(
    batch_size: int = DEFAULT_BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
    cache_path: Path = TRAIN_VAL_CACHE_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Memuat (X_train, y_train, X_val, y_val) ternormalisasi [0,1] tanpa menyentuh TEST."""
    sig = _splits_signature(SPLITS_DIR)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["splits_signature"]) == sig:
                x_tr_u8 = cached["x_train_uint8"]
                y_tr = cached["y_train"].astype(np.int64)
                x_va_u8 = cached["x_val_uint8"]
                y_va = cached["y_val"].astype(np.int64)
                x_tr = normalize_pixels(x_tr_u8)
                x_va = normalize_pixels(x_va_u8)
                return x_tr, y_tr, x_va, y_va, True

    tv_bundle = build_train_val_loaders(
        batch_size=batch_size,
        random_seed=random_seed,
        train_augment=False,
    )
    x_tr, y_tr = tv_bundle.train_loader.load_all_arrays(augment=False)
    x_va, y_va = tv_bundle.val_loader.load_all_arrays(augment=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    x_tr_u8 = np.rint(x_tr * 255.0).astype(np.uint8)
    x_va_u8 = np.rint(x_va * 255.0).astype(np.uint8)
    np.savez(
        cache_path,
        splits_signature=np.array(sig),
        x_train_uint8=x_tr_u8,
        y_train=y_tr.astype(np.int64),
        x_val_uint8=x_va_u8,
        y_val=y_va.astype(np.int64),
    )
    return x_tr, y_tr, x_va, y_va, False


def compute_training_class_weights(
    train_labels: np.ndarray,
    mode: str = "balanced",
) -> dict[int, float] | None:
    """Menghitung class_weight berdasarkan distribusi kelas TRAINING saja."""
    if mode == "none":
        return None

    balanced = compute_balanced_class_weights(train_labels.tolist(), num_classes=NUM_CLASSES)
    if mode == "balanced":
        return balanced
    if mode == "smoothed":
        raw_smooth = {k: float(np.sqrt(v)) for k, v in balanced.items()}
        mean_w = float(np.mean(list(raw_smooth.values())))
        return {k: round(v / mean_w, 6) for k, v in raw_smooth.items()}

    raise ValueError(f"Unsupported class_weight mode '{mode}'. Use 'balanced', 'smoothed', or 'none'.")


def build_tf_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Membungkus tensor ternormalisasi ke tf.data.Dataset yang efisien dan deterministik."""
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), seed=random_seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def evaluate_on_validation(
    model: keras.Model,
    val_ds: tf.data.Dataset,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluasi lengkap + waktu inferensi (1-batch warm-up) pada Validation Set (NO test)."""
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)

    # 1-batch warm-up sebelum timer inferensi dimulai
    _ = model.predict(x_val[:batch_size], batch_size=batch_size, verbose=0)
    t_inf_start = time.perf_counter()
    y_probs = model.predict(val_ds, verbose=0)
    total_inf_sec = time.perf_counter() - t_inf_start
    inf_ms_per_img = (total_inf_sec / len(y_val)) * 1000.0

    y_pred = np.argmax(y_probs, axis=1).astype(np.int64)
    top2_preds = np.argsort(y_probs, axis=1)[:, -2:]
    top2_acc = float(np.mean([y_val[i] in top2_preds[i] for i in range(len(y_val))]))
    metrics = compute_classification_metrics(y_val, y_pred, class_names=CLASS_DISPLAY_NAMES)

    return {
        "val_loss": round(float(val_loss), 6),
        "val_accuracy": round(float(val_acc), 6),
        "val_top2_accuracy": round(top2_acc, 6),
        "val_macro_precision": metrics["macro_avg"]["precision"],
        "val_macro_recall": metrics["macro_avg"]["recall"],
        "val_macro_f1": metrics["macro_avg"]["f1_score"],
        "val_weighted_f1": metrics["weighted_avg"]["f1_score"],
        "val_total_inference_time_sec": round(total_inf_sec, 4),
        "val_inference_time_ms_per_image": round(inf_ms_per_img, 4),
        "per_class": metrics["per_class"],
        "top_confused_pairs": metrics["top_confused_pairs"][:5],
    }


def plot_training_curves(
    history_df: pd.DataFrame,
    best_epoch: int,
    acc_fig_path: Path,
    loss_fig_path: Path,
    title_suffix: str = "FER-2013 Custom CNN",
) -> tuple[Path, Path]:
    """Menyimpan grafik Training vs Validation Accuracy dan Loss pada outputs/figures/."""
    acc_fig_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history_df["epoch"].to_numpy()
    best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })

    # Grafik 1: Accuracy
    fig_acc, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, history_df["accuracy"] * 100, label="Training Accuracy", color="#2A9D8F", linewidth=2.2, marker="o", markersize=3.5)
    ax.plot(epochs, history_df["val_accuracy"] * 100, label="Validation Accuracy", color="#E76F51", linewidth=2.2, marker="s", markersize=3.5)
    ax.axvline(best_epoch, color="#457B9D", linestyle="--", linewidth=1.4, label=f"Best Epoch #{best_epoch} (min val_loss)")
    ax.scatter([best_epoch], [best_row["val_accuracy"] * 100], color="#D62828", s=75, zorder=5)

    text_x = max(1, best_epoch - max(3, len(epochs) // 4))
    text_y = max(5, best_row["val_accuracy"] * 100 - 11)
    ax.annotate(
        f"Best Val Acc: {best_row['val_accuracy'] * 100:.2f}%\n(Epoch {best_epoch})",
        xy=(best_epoch, best_row["val_accuracy"] * 100),
        xytext=(text_x, text_y),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.3"},
    )
    ax.set_title(f"Training Accuracy vs Validation Accuracy — {title_suffix}", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, max(2, int(epochs.max())))
    ax.set_ylim(0, min(100, max(history_df["accuracy"].max(), history_df["val_accuracy"].max()) * 100 + 12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    fig_acc.savefig(acc_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig_acc)

    # Grafik 2: Loss
    fig_loss, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, history_df["loss"], label="Training Loss", color="#457B9D", linewidth=2.2, marker="o", markersize=3.5)
    ax.plot(epochs, history_df["val_loss"], label="Validation Loss", color="#D62828", linewidth=2.2, marker="s", markersize=3.5)
    ax.axvline(best_epoch, color="#2A9D8F", linestyle="--", linewidth=1.4, label=f"Best Epoch #{best_epoch} (val_loss={best_row['val_loss']:.4f})")
    ax.scatter([best_epoch], [best_row["val_loss"]], color="#2A9D8F", s=75, zorder=5)

    if "learning_rate" in history_df.columns and len(history_df) > 1:
        lr_vals = history_df["learning_rate"].to_numpy()
        lr_drop_epochs = [int(epochs[i]) for i in range(1, len(lr_vals)) if lr_vals[i] < lr_vals[i - 1] * 0.99]
        for idx_drop, ep in enumerate(lr_drop_epochs):
            ax.axvline(ep, color="#F4A261", linestyle=":", linewidth=1.2, label="ReduceLROnPlateau (LR × 0.5)" if idx_drop == 0 else None)

    ax.annotate(
        f"Min Val Loss: {best_row['val_loss']:.4f}\n(Epoch {best_epoch})",
        xy=(best_epoch, best_row["val_loss"]),
        xytext=(max(1, best_epoch - max(3, len(epochs) // 4)), best_row["val_loss"] + 0.16),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.3"},
    )
    ax.set_title(f"Training Loss vs Validation Loss — {title_suffix}", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Sparse Categorical Crossentropy)")
    ax.set_xlim(1, max(2, int(epochs.max())))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()
    fig_loss.savefig(loss_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig_loss)

    return acc_fig_path, loss_fig_path


def plot_val_comparison(
    baseline_val: dict[str, Any],
    exp_val: dict[str, Any],
    exp_label: str,
    output_path: Path,
) -> Path:
    """Membuat grafik perbandingan dinamis Baseline vs Eksperimen pada Validation Set (NO test)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [1.0, 1.85]})

    metric_names = ["Val\nAccuracy", "Val Top-2\nAccuracy", "Val\nMacro F1", "Val\nWeighted F1"]
    b_vals = [
        baseline_val["val_accuracy"] * 100,
        baseline_val["val_top2_accuracy"] * 100,
        baseline_val["val_macro_f1"] * 100,
        baseline_val["val_weighted_f1"] * 100,
    ]
    e_vals = [
        exp_val["val_accuracy"] * 100,
        exp_val["val_top2_accuracy"] * 100,
        exp_val["val_macro_f1"] * 100,
        exp_val["val_weighted_f1"] * 100,
    ]

    x0 = np.arange(len(metric_names))
    w = 0.35
    bars_b0 = axes[0].bar(x0 - w / 2, b_vals, w, label="Exp 1: Baseline CNN (111K)", color="#7A93AC", edgecolor="#222222", linewidth=0.6)
    bars_e0 = axes[0].bar(x0 + w / 2, e_vals, w, label=exp_label, color="#2A9D8F", edgecolor="#222222", linewidth=0.6)

    for bars in (bars_b0, bars_e0):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")

    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(metric_names, fontsize=9.5, fontweight="bold")
    axes[0].set_ylabel("Skor (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_title("Perbandingan Metrik Agregat (Validation Set)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.8)

    b_f1s = [r["f1_score"] * 100 for r in baseline_val["per_class"]]
    e_f1s = [r["f1_score"] * 100 for r in exp_val["per_class"]]
    cls_names = [r["class_name"] for r in exp_val["per_class"]]
    x1 = np.arange(len(cls_names))

    bars_b1 = axes[1].bar(x1 - w / 2, b_f1s, w, label="Exp 1: Baseline CNN (balanced)", color="#7A93AC", edgecolor="#222222", linewidth=0.6)
    bars_e1 = axes[1].bar(x1 + w / 2, e_f1s, w, label=exp_label, color="#E76F51", edgecolor="#222222", linewidth=0.6)

    for bars in (bars_b1, bars_e1):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(cls_names, fontsize=9.8, fontweight="bold")
    axes[1].set_ylabel("F1-Score (%)")
    axes[1].set_ylim(0, min(100, max(max(b_f1s), max(e_f1s)) + 15))
    axes[1].set_title("Perbandingan F1-Score per Kelas (Validation Set — 5.742 Citra)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)

    fig.suptitle(f"Evaluasi Validation Set: Baseline CNN vs {exp_label} (Test Set Tetap Terisolasi)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_training_report(
    summary: dict[str, Any],
    report_path: Path,
) -> Path:
    """Menyusun laporan Markdown faktual berdasarkan hasil training, validation & deployment PRD."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = summary["best_epoch_metrics"]
    final = summary["final_epoch_metrics"]
    cfg = summary["config"]
    cw = summary["class_weights"]
    val_eval = summary["validation_evaluation"]
    base_val = summary.get("baseline_validation_comparison")

    cw_lines = (
        "\n".join(f"| `{k}` | `{ID_TO_LABEL[int(k)]}` | `{v:.4f}` |" for k, v in cw.items())
        if cw
        else "| — | Tidak digunakan (`none`) | — |"
    )

    per_class_section = "\n".join(
        f"| `{r['label_id']}` | {r['class_name']} | {r['precision'] * 100:.2f}% | "
        f"{r['recall'] * 100:.2f}% | {r['f1_score'] * 100:.2f}% | {r['support']:,} |"
        for r in val_eval["per_class"]
    )

    comparison_block = ""
    if base_val is not None:
        d_acc = (val_eval["val_accuracy"] - base_val["val_accuracy"]) * 100
        d_loss = val_eval["val_loss"] - base_val["val_loss"]
        d_mf1 = (val_eval["val_macro_f1"] - base_val["val_macro_f1"]) * 100
        d_wf1 = (val_eval["val_weighted_f1"] - base_val["val_weighted_f1"]) * 100
        d_top2 = (val_eval["val_top2_accuracy"] - base_val["val_top2_accuracy"]) * 100

        cls_cmp_rows = "\n".join(
            f"| `{e['label_id']}` | {e['class_name']} | {b['f1_score'] * 100:.2f}% | "
            f"**{e['f1_score'] * 100:.2f}%** | {(e['f1_score'] - b['f1_score']) * 100:+.2f} pp | "
            f"{b['recall'] * 100:.2f}% → {e['recall'] * 100:.2f}% |"
            for b, e in zip(base_val["per_class"], val_eval["per_class"])
        )

        comparison_block = f"""
---

## 4. Perbandingan Head-to-Head pada Validation Set (5.742 Citra, Tanpa Test)

| Metrik (Validation Set) | Exp 1: Baseline CNN (`111K`, `balanced`) | `{summary['experiment'].upper()}` (`{cfg['model_variant']}`, `{cfg['class_weight_mode']}`) | Delta |
|---|---:|---:|---:|
| **Validation Accuracy** | {base_val['val_accuracy'] * 100:.2f}% | **{val_eval['val_accuracy'] * 100:.2f}%** | `{d_acc:+.2f} pp` |
| **Validation Loss** | {base_val['val_loss']:.4f} | **{val_eval['val_loss']:.4f}** | `{d_loss:+.4f}` |
| **Validation Top-2 Accuracy** | {base_val['val_top2_accuracy'] * 100:.2f}% | **{val_eval['val_top2_accuracy'] * 100:.2f}%** | `{d_top2:+.2f} pp` |
| **Validation Macro F1-Score** | {base_val['val_macro_f1'] * 100:.2f}% | **{val_eval['val_macro_f1'] * 100:.2f}%** | `{d_mf1:+.2f} pp` |
| **Validation Weighted F1-Score** | {base_val['val_weighted_f1'] * 100:.2f}% | **{val_eval['val_weighted_f1'] * 100:.2f}%** | `{d_wf1:+.2f} pp` |

### Perbandingan F1-Score & Recall per Kelas (Validation Set)

| ID | Kelas | Baseline Val F1 | `{summary['experiment'].upper()}` Val F1 | Delta F1 | Recall (Base → Baru) |
|---:|---|---:|---:|---:|---:|
{cls_cmp_rows}
"""

    md = f"""# Laporan Model Training — {summary['experiment'].upper()} ({summary['model_name']})

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013)  
**Eksperimen:** `{summary['experiment']}` (Varian: `{cfg['model_variant']}`, `{summary['model_params']['total']:,}` parameter, `{summary['model_size_mb']:.4f} MB`)  
**Waktu Selesai (UTC):** `{summary['completed_at_utc']}`

---

## 1. Konfigurasi, Isolasi Data & Metrik Deployment PRD

| Parameter | Nilai |
|---|---|
| Data Training (`model.fit`) | `{summary['data_counts']['train']:,}` citra (`data/splits/train.csv`) |
| Data Validation (`validation_data`) | `{summary['data_counts']['val']:,}` citra (`data/splits/val.csv`) |
| Data Testing selama Training & Seleksi | **Tidak diakses / tidak digunakan (`0` citra)** |
| Input Shape | `{cfg['input_shape']}` (grayscale 1 kanal, normalisasi tunggal `[0, 1]`) |
| Total / Trainable / Non-Trainable Params | `{summary['model_params']['total']:,}` / `{summary['model_params']['trainable']:,}` / `{summary['model_params']['non_trainable']:,}` |
| **Model Size (`model_size_mb`)** | **`{summary['model_size_mb']:.4f} MB`** |
| **Val Inference Time (`val_inference_time_ms_per_image`)** | **`{summary['val_inference_time_ms_per_image']:.4f} ms/citra`** (`{val_eval['val_total_inference_time_sec']:.3f}s` total, setelah 1-batch warm-up) |
| Max Epochs / Batch Size | `{cfg['max_epochs']}` / `{cfg['batch_size']}` |
| Optimizer & Initial LR | `Adam` (`lr = {cfg['learning_rate']}`) |
| Loss & Metric | `SparseCategoricalCrossentropy` / `accuracy` |
| Class Weight Mode | `{cfg['class_weight_mode']}` (khusus `TRAIN`) |
| Callbacks | `EarlyStopping(patience={cfg['early_stopping_patience']})`, `ReduceLROnPlateau(factor={cfg['reduce_lr_factor']}, patience={cfg['reduce_lr_patience']})`, `ModelCheckpoint(val_loss)` |
| Random Seed & Device | `{cfg['random_seed']}` / `{summary['device']}` |

---

## 2. Ringkasan Hasil Training Aktual

| Metrik | Best Epoch (`#{best['epoch']}`, min `val_loss`) | Final Epoch (`#{final['epoch']}`) |
|---|---:|---:|
| **Training Accuracy** | **{best['accuracy'] * 100:.2f}%** (`{best['accuracy']:.4f}`) | {final['accuracy'] * 100:.2f}% (`{final['accuracy']:.4f}`) |
| **Validation Accuracy** | **{best['val_accuracy'] * 100:.2f}%** (`{best['val_accuracy']:.4f}`) | {final['val_accuracy'] * 100:.2f}% (`{final['val_accuracy']:.4f}`) |
| **Training Loss** | **{best['loss']:.4f}** | {final['loss']:.4f} |
| **Validation Loss** | **{best['val_loss']:.4f}** | {final['val_loss']:.4f} |
| **Learning Rate** | `{best['learning_rate']:.2e}` | `{final['learning_rate']:.2e}` |

- **Total Epoch Dijalankan:** `{summary['epochs_executed']}` / `{cfg['max_epochs']}` (`early_stopped = {summary['early_stopped']}`).
- **Total Waktu Training:** `{summary['training_time_formatted']}` (`{summary['avg_seconds_per_epoch']:.2f}` detik/epoch).

---

## 3. Evaluasi Detail pada Validation Set (`5.742` Citra)

- **Val Accuracy:** **{val_eval['val_accuracy'] * 100:.2f}%** | **Val Loss:** `{val_eval['val_loss']:.4f}` | **Val Top-2 Acc:** `{val_eval['val_top2_accuracy'] * 100:.2f}%`
- **Val Macro Precision:** `{val_eval['val_macro_precision'] * 100:.2f}%` | **Val Macro Recall:** `{val_eval['val_macro_recall'] * 100:.2f}%`
- **Val Macro F1:** **{val_eval['val_macro_f1'] * 100:.2f}%** | **Val Weighted F1:** **{val_eval['val_weighted_f1'] * 100:.2f}%**

| ID | Kelas | Val Precision | Val Recall | Val F1-Score | Support |
|---:|---|---:|---:|---:|---:|
{per_class_section}

### Bobot Kelas Training (`class_weight_mode = {cfg['class_weight_mode']}`)

| `label_id` | Kelas | Bobot |
|---:|---|---:|
{cw_lines}
{comparison_block}

---

## 5. Artefak Tersimpan

- **Model Terbaik:** `{summary['artifacts']['best_model']}`
- **Checkpoint Terakhir (`--resume`):** `{summary['artifacts']['last_model']}`
- **History Training:** `{summary['artifacts']['history_csv']}` & `{summary['artifacts']['history_json']}`
- **Grafik:** `{summary['artifacts']['accuracy_figure']}`, `{summary['artifacts']['loss_figure']}`{f", `{summary['artifacts']['val_compare_figure']}`" if summary['artifacts'].get('val_compare_figure') else ""}
"""
    report_path.write_text(md, encoding="utf-8")
    return report_path


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(seconds, 60.0)
    hours, mins = divmod(mins, 60.0)
    if hours >= 1:
        return f"{int(hours)} jam {int(mins)} menit {secs:.1f} detik"
    if mins >= 1:
        return f"{int(mins)} menit {secs:.1f} detik"
    return f"{secs:.1f} detik"


def train(
    experiment: str = "baseline",
    model_variant: str = "baseline",
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    class_weight_mode: str = "balanced",
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    reduce_lr_patience: int = REDUCE_LR_PATIENCE,
    random_seed: int = RANDOM_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    """Menjalankan proses training model secara penuh dengan isolasi artefak per eksperimen."""
    device_str = configure_gpu()
    keras.utils.set_random_seed(random_seed)

    paths = resolve_experiment_paths(experiment)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== EKSPERIMEN '{experiment}' (variant='{model_variant}') | TEST DIISOLASI ===", flush=True)
    t_load_start = time.perf_counter()
    x_train, y_train, x_val, y_val, from_cache = load_train_val_arrays_cached(
        batch_size=batch_size,
        random_seed=random_seed,
    )
    load_elapsed = time.perf_counter() - t_load_start

    assert x_train.shape[1:] == INPUT_SHAPE and x_val.shape[1:] == INPUT_SHAPE
    assert x_train.dtype == np.float32 and x_val.dtype == np.float32
    assert 0.0 <= float(x_train.min()) <= float(x_train.max()) <= 1.0
    assert 0.0 <= float(x_val.min()) <= float(x_val.max()) <= 1.0

    print(
        f"Data dimuat dalam {load_elapsed:.2f}s (from_cache={from_cache}) | "
        f"Train={x_train.shape} | Val={x_val.shape}",
        flush=True,
    )

    class_weights = compute_training_class_weights(y_train, mode=class_weight_mode)
    print(f"Class weight mode: {class_weight_mode} -> {class_weights}", flush=True)

    train_ds, val_ds = build_tf_datasets(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,
        random_seed=random_seed,
    )

    prior_history_df: pd.DataFrame | None = None
    initial_epoch = 0
    if resume and paths["last_model"].exists() and paths["history_csv"].exists():
        print(f"Melanjutkan training dari checkpoint: {paths['last_model']}", flush=True)
        model = keras.models.load_model(paths["last_model"])
        prior_history_df = pd.read_csv(paths["history_csv"])
        initial_epoch = int(prior_history_df["epoch"].max())
        if initial_epoch >= max_epochs:
            raise ValueError(
                f"Training sudah mencapai epoch {initial_epoch} >= max_epochs ({max_epochs})."
            )
    else:
        model = create_model_by_variant(
            model_variant=model_variant,
            seed=random_seed,
            use_data_augmentation=True,
        )
        model = compile_model(model, learning_rate=learning_rate)

    total_params = int(model.count_params())
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    print(
        f"Model: {model.name} | Total params: {total_params:,} "
        f"(Trainable: {trainable_params:,}, Non-trainable: {non_trainable_params:,})",
        flush=True,
    )

    progress_cb = EpochProgressCallback(total_epochs=max_epochs)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=reduce_lr_patience,
        min_lr=REDUCE_LR_MIN_LR,
        verbose=1,
    )
    best_ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(paths["best_model"]),
        monitor="val_loss",
        save_best_only=True,
        initial_value_threshold=(
            float(prior_history_df["val_loss"].min()) if prior_history_df is not None else None
        ),
        verbose=1,
    )
    last_ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(paths["last_model"]),
        save_best_only=False,
        verbose=0,
    )

    print(
        f"\n=== MULAI TRAINING ({device_str}) | "
        f"Epochs={initial_epoch + 1}..{max_epochs} | Batch={batch_size} | LR={learning_rate} ===",
        flush=True,
    )
    t_train_start = time.perf_counter()

    # shuffle=False karena train_ds sudah di-shuffle melalui tf.data.Dataset.shuffle()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        shuffle=False,
        callbacks=[early_stopping_cb, reduce_lr_cb, best_ckpt_cb, last_ckpt_cb, progress_cb],
        verbose=0,
    )

    training_elapsed = time.perf_counter() - t_train_start

    hist_dict = history.history
    n_new = len(hist_dict["loss"])
    lr_key = "learning_rate" if "learning_rate" in hist_dict else ("lr" if "lr" in hist_dict else None)
    lr_series = (
        [float(v) for v in hist_dict[lr_key]]
        if lr_key is not None
        else [float(model.optimizer.learning_rate.numpy())] * n_new
    )

    new_history_df = pd.DataFrame({
        "epoch": list(range(initial_epoch + 1, initial_epoch + n_new + 1)),
        "loss": [float(v) for v in hist_dict["loss"]],
        "accuracy": [float(v) for v in hist_dict["accuracy"]],
        "val_loss": [float(v) for v in hist_dict["val_loss"]],
        "val_accuracy": [float(v) for v in hist_dict["val_accuracy"]],
        "learning_rate": lr_series,
        "epoch_time_sec": progress_cb.epoch_times[:n_new],
    })
    history_df = (
        pd.concat([prior_history_df, new_history_df], ignore_index=True)
        if prior_history_df is not None
        else new_history_df
    )

    best_idx = int(history_df["val_loss"].idxmin())
    best_row = history_df.iloc[best_idx]
    final_row = history_df.iloc[-1]
    best_epoch = int(best_row["epoch"])
    epochs_executed = int(len(history_df))
    early_stopped = epochs_executed < max_epochs

    # Evaluasi mendalam + metrik deployment PRD pada VALIDATION SET (TEST tetap 100% terisolasi!)
    best_loaded_model = keras.models.load_model(paths["best_model"])
    model_size_mb = round(os.path.getsize(paths["best_model"]) / (1024 ** 2), 4)
    val_eval = evaluate_on_validation(
        best_loaded_model,
        val_ds=val_ds,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,
    )

    # Perbandingan langsung pada VALIDATION SET terhadap Baseline jika bukan baseline
    baseline_val_eval: dict[str, Any] | None = None
    val_compare_fig: Path | None = None
    if experiment != "baseline" and BEST_MODEL_PATH.exists():
        print("\nMengevaluasi Baseline CNN pada Validation Set yang sama untuk perbandingan adil...", flush=True)
        baseline_model = keras.models.load_model(BEST_MODEL_PATH)
        baseline_val_eval = evaluate_on_validation(
            baseline_model,
            val_ds=val_ds,
            x_val=x_val,
            y_val=y_val,
            batch_size=batch_size,
        )
        variant_name = VARIANT_DISPLAY_NAMES.get(model_variant, model.name)
        param_str = f"{total_params / 1_000_000:.2f}M" if total_params >= 1_000_000 else f"{total_params // 1000}K"
        exp_label = f"{experiment.upper()}: {variant_name} ({param_str}, {class_weight_mode})"
        val_compare_fig = plot_val_comparison(
            baseline_val=baseline_val_eval,
            exp_val=val_eval,
            exp_label=exp_label,
            output_path=paths["val_compare_figure"],
        )

    history_df.to_csv(paths["history_csv"], index=False)
    title_suffix = (
        f"FER-2013 {experiment.upper()} ({model.name})"
        if experiment != "baseline"
        else "FER-2013 Custom CNN"
    )
    acc_fig_path, loss_fig_path = plot_training_curves(
        history_df=history_df,
        best_epoch=best_epoch,
        acc_fig_path=paths["acc_figure"],
        loss_fig_path=paths["loss_figure"],
        title_suffix=title_suffix,
    )

    artifacts = {
        "best_model": paths["best_model"].relative_to(PROJECT_ROOT).as_posix(),
        "last_model": paths["last_model"].relative_to(PROJECT_ROOT).as_posix(),
        "history_csv": paths["history_csv"].relative_to(PROJECT_ROOT).as_posix(),
        "history_json": paths["history_json"].relative_to(PROJECT_ROOT).as_posix(),
        "training_report": paths["report_md"].relative_to(PROJECT_ROOT).as_posix(),
        "accuracy_figure": acc_fig_path.relative_to(PROJECT_ROOT).as_posix(),
        "loss_figure": loss_fig_path.relative_to(PROJECT_ROOT).as_posix(),
        "val_compare_figure": val_compare_fig.relative_to(PROJECT_ROOT).as_posix() if val_compare_fig else None,
    }

    summary: dict[str, Any] = {
        "experiment": experiment,
        "model_name": model.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_str,
        "tensorflow_version": tf.__version__,
        "resumed_from_epoch": initial_epoch,
        "epochs_executed": epochs_executed,
        "early_stopped": early_stopped,
        "training_time_seconds": round(training_elapsed, 2),
        "training_time_formatted": _format_duration(training_elapsed),
        "avg_seconds_per_epoch": round(training_elapsed / max(1, n_new), 2),
        "model_params": {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        },
        "model_size_mb": model_size_mb,
        "val_inference_time_ms_per_image": val_eval["val_inference_time_ms_per_image"],
        "val_total_inference_time_sec": val_eval["val_total_inference_time_sec"],
        "data_counts": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test_used_in_training": 0,
        },
        "config": {
            "model_variant": model_variant,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": "SparseCategoricalCrossentropy",
            "metrics": ["accuracy"],
            "input_shape": list(INPUT_SHAPE),
            "num_classes": NUM_CLASSES,
            "label_mapping": LABEL_TO_ID,
            "class_weight_mode": class_weight_mode,
            "early_stopping_patience": early_stopping_patience,
            "reduce_lr_factor": REDUCE_LR_FACTOR,
            "reduce_lr_patience": reduce_lr_patience,
            "reduce_lr_min_lr": REDUCE_LR_MIN_LR,
            "random_seed": random_seed,
            "in_model_augmentation_training_only": True,
        },
        "class_weights": {str(k): v for k, v in class_weights.items()} if class_weights else None,
        "best_epoch_metrics": {
            "epoch": best_epoch,
            "loss": round(float(best_row["loss"]), 6),
            "accuracy": round(float(best_row["accuracy"]), 6),
            "val_loss": round(float(best_row["val_loss"]), 6),
            "val_accuracy": round(float(best_row["val_accuracy"]), 6),
            "learning_rate": float(best_row["learning_rate"]),
        },
        "final_epoch_metrics": {
            "epoch": int(final_row["epoch"]),
            "loss": round(float(final_row["loss"]), 6),
            "accuracy": round(float(final_row["accuracy"]), 6),
            "val_loss": round(float(final_row["val_loss"]), 6),
            "val_accuracy": round(float(final_row["val_accuracy"]), 6),
            "learning_rate": float(final_row["learning_rate"]),
        },
        "best_model_verified_val_loss": val_eval["val_loss"],
        "best_model_verified_val_accuracy": val_eval["val_accuracy"],
        "validation_evaluation": val_eval,
        "baseline_validation_comparison": baseline_val_eval,
        "artifacts": artifacts,
        "history": history_df.to_dict(orient="list"),
    }

    paths["history_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_training_report(summary=summary, report_path=paths["report_md"])

    print(f"\n================ RINGKASAN HASIL TRAINING ({experiment.upper()}) ================")
    print(f"Model                   : {model.name} ({total_params:,} params | {model_size_mb:.4f} MB)")
    print(f"Val Inference Time      : {val_eval['val_inference_time_ms_per_image']:.4f} ms/citra ({val_eval['val_total_inference_time_sec']:.3f}s total)")
    print(f"Epoch yang dijalankan   : {epochs_executed} / {max_epochs} (best epoch = {best_epoch}, early_stopped = {early_stopped})")
    print(f"Total waktu training    : {summary['training_time_formatted']} ({summary['avg_seconds_per_epoch']:.2f} s/epoch)")
    print(f"Best Epoch #{best_epoch:02d} Metrics :")
    print(f"  Training Accuracy     : {best_row['accuracy'] * 100:.2f}% ({best_row['accuracy']:.4f})")
    print(f"  Validation Accuracy   : {best_row['val_accuracy'] * 100:.2f}% ({best_row['val_accuracy']:.4f})")
    print(f"  Training Loss         : {best_row['loss']:.4f}")
    print(f"  Validation Loss       : {best_row['val_loss']:.4f}")
    print(
        f"Validation Detailed     : Macro P={val_eval['val_macro_precision'] * 100:.2f}% | "
        f"Macro R={val_eval['val_macro_recall'] * 100:.2f}% | "
        f"Macro F1={val_eval['val_macro_f1'] * 100:.2f}% | "
        f"Weighted F1={val_eval['val_weighted_f1'] * 100:.2f}% | "
        f"Top-2={val_eval['val_top2_accuracy'] * 100:.2f}%"
    )
    if baseline_val_eval is not None:
        print(
            f"Vs Baseline (VAL)       : Acc {baseline_val_eval['val_accuracy']*100:.2f}% -> {val_eval['val_accuracy']*100:.2f}% "
            f"({(val_eval['val_accuracy']-baseline_val_eval['val_accuracy'])*100:+.2f} pp) | "
            f"Macro F1 {baseline_val_eval['val_macro_f1']*100:.2f}% -> {val_eval['val_macro_f1']*100:.2f}% "
            f"({(val_eval['val_macro_f1']-baseline_val_eval['val_macro_f1'])*100:+.2f} pp)"
        )
    print(f"Model terbaik tersimpan : {paths['best_model']}")
    print(f"Laporan tersimpan       : {paths['report_md']}")
    print("===================================================================")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FER-2013 Deep Learning models (Train & Validation only).")
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name (e.g., baseline, exp2, mlp, mobilenetv2)")
    parser.add_argument("--model-variant", choices=sorted(MODEL_REGISTRY.keys()), default="baseline", help="Model architecture variant")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Max epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--class-weight-mode", choices=["balanced", "smoothed", "none"], default="balanced", help="Class weight mode")
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE, help="EarlyStopping patience")
    parser.add_argument("--reduce-lr-patience", type=int, default=REDUCE_LR_PATIENCE, help="ReduceLROnPlateau patience")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--resume", action="store_true", help="Resume training from last saved checkpoint")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        experiment=args.experiment,
        model_variant=args.model_variant,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        class_weight_mode=args.class_weight_mode,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        random_seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
