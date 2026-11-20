"""Model Training Pipeline for FER-2013 Custom CNN (TensorFlow & Keras).

Melatih model Custom CNN dari ``src/model.py`` menggunakan data terproses dari
``src/data_loader.py`` dan ``src/preprocessing.py``.

Jaminan Metodologis:
1. HANYA subset Training (22.967 citra) dan Validation (5.742 citra) yang dimuat.
   Subset Testing (``data/raw/test/`` / ``data/splits/test.csv``) TIDAK pernah
   diakses ataupun digunakan selama proses training.
2. Normalisasi piksel [0, 255] -> [0.0, 1.0] dilakukan tepat SATU kali oleh
   ``src/preprocessing.py`` (dilindungi anti-double-normalization guard).
3. Data augmentation ringan aktif HANYA di dalam model saat ``training=True``
   (``train_augment=False`` pada loader agar tidak terjadi augmentasi ganda;
   validation selalu dievaluasi tanpa augmentasi).
4. Mendukung resumable training (``--resume``) dengan konfigurasi yang konsisten.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    FIGURES_DIR,
    ID_TO_LABEL,
    INPUT_SHAPE,
    LABEL_TO_ID,
    LAST_MODEL_PATH,
    MAX_EPOCHS,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_LR,
    REDUCE_LR_PATIENCE,
    REPORTS_DIR,
    TRAINING_HISTORY_CSV,
    TRAINING_HISTORY_JSON,
    TRAINING_REPORT_MD,
)
from src.data_loader import build_train_val_loaders
from src.model import build_model, compile_model
from src.preprocessing import compute_balanced_class_weights


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


def compute_training_class_weights(
    train_labels: np.ndarray,
    mode: str = "balanced",
) -> dict[int, float] | None:
    """Menghitung class_weight berdasarkan distribusi kelas TRAINING saja.

    Args:
        train_labels: Array label integer (0..6) khusus subset training.
        mode:
            - ``"balanced"``: bobot standar N / (K * n_c) dari ``src.preprocessing``.
            - ``"smoothed"``: bobot lunak sqrt(N / (K * n_c)), dinormalisasi mean=1.0.
            - ``"none"``: tanpa pembobotan kelas (mengembalikan ``None``).
    """
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


def plot_training_curves(
    history_df: pd.DataFrame,
    best_epoch: int,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Menyimpan grafik Training vs Validation Accuracy dan Loss pada outputs/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)
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

    # ------------------------------------------------------------------
    # Grafik 1: Training Accuracy vs Validation Accuracy
    # ------------------------------------------------------------------
    fig_acc, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        epochs,
        history_df["accuracy"] * 100,
        label="Training Accuracy",
        color="#2A9D8F",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    ax.plot(
        epochs,
        history_df["val_accuracy"] * 100,
        label="Validation Accuracy",
        color="#E76F51",
        linewidth=2.2,
        marker="s",
        markersize=3.5,
    )
    ax.axvline(
        best_epoch,
        color="#457B9D",
        linestyle="--",
        linewidth=1.4,
        label=f"Best Epoch #{best_epoch} (min val_loss)",
    )
    ax.scatter(
        [best_epoch],
        [best_row["val_accuracy"] * 100],
        color="#D62828",
        s=75,
        zorder=5,
    )
    ax.annotate(
        f"Best Val Acc: {best_row['val_accuracy'] * 100:.2f}%\n(Epoch {best_epoch})",
        xy=(best_epoch, best_row["val_accuracy"] * 100),
        xytext=(max(1, best_epoch - max(2, len(epochs) // 5)), max(5, best_row["val_accuracy"] * 100 - 10)),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.3"},
    )
    ax.set_title("Training Accuracy vs Validation Accuracy — FER-2013 Custom CNN", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, max(2, int(epochs.max())))
    ax.set_ylim(0, min(100, max(history_df["accuracy"].max(), history_df["val_accuracy"].max()) * 100 + 12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()

    acc_fig_path = figures_dir / "08_train_val_accuracy.png"
    fig_acc.savefig(acc_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig_acc)

    # ------------------------------------------------------------------
    # Grafik 2: Training Loss vs Validation Loss
    # ------------------------------------------------------------------
    fig_loss, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        epochs,
        history_df["loss"],
        label="Training Loss",
        color="#457B9D",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    ax.plot(
        epochs,
        history_df["val_loss"],
        label="Validation Loss",
        color="#D62828",
        linewidth=2.2,
        marker="s",
        markersize=3.5,
    )
    ax.axvline(
        best_epoch,
        color="#2A9D8F",
        linestyle="--",
        linewidth=1.4,
        label=f"Best Epoch #{best_epoch} (val_loss={best_row['val_loss']:.4f})",
    )
    ax.scatter(
        [best_epoch],
        [best_row["val_loss"]],
        color="#2A9D8F",
        s=75,
        zorder=5,
    )

    # Tandai titik penurunan learning rate (ReduceLROnPlateau) jika terjadi
    if "learning_rate" in history_df.columns and len(history_df) > 1:
        lr_vals = history_df["learning_rate"].to_numpy()
        lr_drop_epochs = [
            int(epochs[i]) for i in range(1, len(lr_vals)) if lr_vals[i] < lr_vals[i - 1] * 0.99
        ]
        for idx_drop, ep in enumerate(lr_drop_epochs):
            ax.axvline(
                ep,
                color="#F4A261",
                linestyle=":",
                linewidth=1.2,
                label="ReduceLROnPlateau (LR × 0.5)" if idx_drop == 0 else None,
            )

    ax.annotate(
        f"Min Val Loss: {best_row['val_loss']:.4f}\n(Epoch {best_epoch})",
        xy=(best_epoch, best_row["val_loss"]),
        xytext=(max(1, best_epoch - max(2, len(epochs) // 5)), best_row["val_loss"] + 0.18),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.3"},
    )
    ax.set_title("Training Loss vs Validation Loss — FER-2013 Custom CNN", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Sparse Categorical Crossentropy)")
    ax.set_xlim(1, max(2, int(epochs.max())))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()

    loss_fig_path = figures_dir / "09_train_val_loss.png"
    fig_loss.savefig(loss_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig_loss)

    return acc_fig_path, loss_fig_path


def write_training_report(
    summary: dict[str, Any],
    history_df: pd.DataFrame,
    report_path: Path = TRAINING_REPORT_MD,
) -> Path:
    """Menyusun laporan Markdown faktual berdasarkan hasil training aktual."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = summary["best_epoch_metrics"]
    final = summary["final_epoch_metrics"]
    cfg = summary["config"]
    cw = summary["class_weights"]

    cw_lines = (
        "\n".join(
            f"| `{k}` | `{ID_TO_LABEL[int(k)]}` | `{v:.4f}` |"
            for k, v in cw.items()
        )
        if cw
        else "| — | Tidak digunakan (`none`) | — |"
    )

    md = f"""# Laporan Model Training — FER-2013 Custom CNN

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Tahap:** Tahap 5 — Model Training  
**Skrip Training:** `src/train.py`  
**Waktu Selesai (UTC):** `{summary['completed_at_utc']}`

---

## 1. Konfigurasi & Isolasi Data

| Parameter | Nilai |
|---|---|
| Data Training (`model.fit`) | `{summary['data_counts']['train']:,}` citra (`data/splits/train.csv`) |
| Data Validation (`validation_data`) | `{summary['data_counts']['val']:,}` citra (`data/splits/val.csv`) |
| Data Testing selama Training | **Tidak diakses / tidak digunakan (`0` citra)** |
| Input Shape | `{cfg['input_shape']}` (grayscale 1 kanal, normalisasi tunggal `[0, 1]`) |
| Augmentasi Data | Aktif hanya saat `training=True` di dalam model (`RandomFlip`, `RandomRotation ±10°`, `RandomZoom ±10%`) |
| Max Epochs | `{cfg['max_epochs']}` |
| Batch Size | `{cfg['batch_size']}` |
| Optimizer & Initial LR | `Adam` (`lr = {cfg['learning_rate']}`) |
| Loss Function | `SparseCategoricalCrossentropy` |
| Metric | `accuracy` |
| Class Weight Mode | `{cfg['class_weight_mode']}` (dihitung khusus dari distribusi `TRAIN`) |
| Callbacks | `EarlyStopping(val_loss, patience={cfg['early_stopping_patience']}, restore_best_weights=True)`, `ReduceLROnPlateau(val_loss, factor={cfg['reduce_lr_factor']}, patience={cfg['reduce_lr_patience']})`, `ModelCheckpoint(val_loss, save_best_only=True)` |
| Random Seed | `{cfg['random_seed']}` |
| Perangkat Eksekusi | `{summary['device']}` |

---

## 2. Ringkasan Hasil Training Aktual

| Metrik | Best Epoch (`#{best['epoch']}`, min `val_loss`) | Final Epoch (`#{final['epoch']}`) |
|---|---:|---:|
| **Training Accuracy** | **{best['accuracy'] * 100:.2f}%** (`{best['accuracy']:.4f}`) | {final['accuracy'] * 100:.2f}% (`{final['accuracy']:.4f}`) |
| **Validation Accuracy** | **{best['val_accuracy'] * 100:.2f}%** (`{best['val_accuracy']:.4f}`) | {final['val_accuracy'] * 100:.2f}% (`{final['val_accuracy']:.4f}`) |
| **Training Loss** | **{best['loss']:.4f}** | {final['loss']:.4f} |
| **Validation Loss** | **{best['val_loss']:.4f}** | {final['val_loss']:.4f} |
| **Learning Rate** | `{best['learning_rate']:.2e}` | `{final['learning_rate']:.2e}` |

- **Total Epoch Dijalankan:** `{summary['epochs_executed']}` dari maksimal `{cfg['max_epochs']}` (`early_stopped = {summary['early_stopped']}`).
- **Total Waktu Training:** `{summary['training_time_formatted']}` (`{summary['training_time_seconds']:.2f}` detik; rata-rata `{summary['avg_seconds_per_epoch']:.2f}` detik/epoch).
- **Verifikasi `model.evaluate(val)` dengan bobot terbaik tersimpan (`models/fer2013_cnn.keras`):** `val_loss = {summary['best_model_verified_val_loss']:.4f}`, `val_accuracy = {summary['best_model_verified_val_accuracy'] * 100:.2f}%`.

---

## 3. Bobot Kelas (`class_weight`, Khusus Training)

| `label_id` | Kelas | Bobot |
|---:|---|---:|
{cw_lines}

---

## 4. Artefak Tersimpan

- **Model Terbaik:** `{summary['artifacts']['best_model']}`
- **Checkpoint Epoch Terakhir (untuk `--resume`):** `{summary['artifacts']['last_model']}`
- **History Training (CSV & JSON):** `{summary['artifacts']['history_csv']}`, `{summary['artifacts']['history_json']}`
- **Grafik Accuracy:** `{summary['artifacts']['accuracy_figure']}`
- **Grafik Loss:** `{summary['artifacts']['loss_figure']}`
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
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    class_weight_mode: str = "balanced",
    random_seed: int = RANDOM_SEED,
    resume: bool = False,
    best_model_path: Path = BEST_MODEL_PATH,
    last_model_path: Path = LAST_MODEL_PATH,
) -> dict[str, Any]:
    """Menjalankan proses training Custom CNN secara penuh dengan konfigurasi konsisten."""
    keras.utils.set_random_seed(random_seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    device_str = f"GPU ({len(gpus)} device)" if gpus else "CPU (TensorFlow oneDNN)"

    print("=== PERSIAPAN DATA TRAINING & VALIDATION (TEST DIISOLASI) ===", flush=True)
    t_load_start = time.perf_counter()

    # PENTING: build_train_val_loaders() hanya memuat train.csv & val.csv (NO test!)
    # train_augment=False karena augmentasi ringan aktif di dalam model (training=True)
    tv_bundle = build_train_val_loaders(
        batch_size=batch_size,
        random_seed=random_seed,
        train_augment=False,
    )
    x_train, y_train = tv_bundle.train_loader.load_all_arrays(augment=False)
    x_val, y_val = tv_bundle.val_loader.load_all_arrays(augment=False)
    load_elapsed = time.perf_counter() - t_load_start

    # Validasi ketat pra-training
    assert x_train.shape == (len(tv_bundle.train_df), *INPUT_SHAPE), f"Shape train salah: {x_train.shape}"
    assert x_val.shape == (len(tv_bundle.val_df), *INPUT_SHAPE), f"Shape val salah: {x_val.shape}"
    assert x_train.dtype == np.float32 and x_val.dtype == np.float32
    assert 0.0 <= float(x_train.min()) <= float(x_train.max()) <= 1.0
    assert 0.0 <= float(x_val.min()) <= float(x_val.max()) <= 1.0

    print(
        f"Data dimuat dalam {load_elapsed:.1f}s | "
        f"Train={x_train.shape} (range [{x_train.min():.2f}, {x_train.max():.2f}]) | "
        f"Val={x_val.shape} (range [{x_val.min():.2f}, {x_val.max():.2f}])",
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

    # Dukungan melanjutkan training (--resume) secara konsisten
    prior_history_df: pd.DataFrame | None = None
    initial_epoch = 0
    if resume and last_model_path.exists() and TRAINING_HISTORY_CSV.exists():
        print(f"Melanjutkan training dari checkpoint: {last_model_path}", flush=True)
        model = keras.models.load_model(last_model_path)
        prior_history_df = pd.read_csv(TRAINING_HISTORY_CSV)
        initial_epoch = int(prior_history_df["epoch"].max())
        if initial_epoch >= max_epochs:
            raise ValueError(
                f"Training sudah mencapai epoch {initial_epoch} >= max_epochs ({max_epochs}). "
                "Naikkan --epochs untuk melanjutkan."
            )
    else:
        model = build_model(use_data_augmentation=True, seed=random_seed)
        model = compile_model(model, learning_rate=learning_rate)

    # Callbacks wajib sesuai spesifikasi
    progress_cb = EpochProgressCallback(total_epochs=max_epochs)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN_LR,
        verbose=1,
    )
    best_ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        initial_value_threshold=(
            float(prior_history_df["val_loss"].min()) if prior_history_df is not None else None
        ),
        verbose=1,
    )
    last_ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(last_model_path),
        save_best_only=False,
        verbose=0,
    )

    print(
        f"\n=== MULAI TRAINING ({device_str}) | "
        f"Epochs={initial_epoch + 1}..{max_epochs} | Batch={batch_size} | LR={learning_rate} ===",
        flush=True,
    )
    t_train_start = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=[early_stopping_cb, reduce_lr_cb, best_ckpt_cb, last_ckpt_cb, progress_cb],
        verbose=0,  # Progress rapi setiap epoch ditampilkan oleh EpochProgressCallback
    )

    training_elapsed = time.perf_counter() - t_train_start

    # Susun DataFrame history aktual (digabung dengan history sebelumnya jika --resume)
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

    if prior_history_df is not None:
        history_df = pd.concat([prior_history_df, new_history_df], ignore_index=True)
    else:
        history_df = new_history_df

    # Identifikasi epoch terbaik berdasarkan minimum val_loss
    best_idx = int(history_df["val_loss"].idxmin())
    best_row = history_df.iloc[best_idx]
    final_row = history_df.iloc[-1]
    best_epoch = int(best_row["epoch"])
    epochs_executed = int(len(history_df))
    early_stopped = epochs_executed < max_epochs

    # Verifikasi model terbaik yang tersimpan pada disk (hanya pada validation set!)
    best_loaded_model = keras.models.load_model(best_model_path)
    verified_val_loss, verified_val_acc = best_loaded_model.evaluate(val_ds, verbose=0)

    # Simpan history CSV & grafik
    history_df.to_csv(TRAINING_HISTORY_CSV, index=False)
    acc_fig_path, loss_fig_path = plot_training_curves(
        history_df=history_df,
        best_epoch=best_epoch,
        figures_dir=FIGURES_DIR,
    )

    summary: dict[str, Any] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_str,
        "tensorflow_version": tf.__version__,
        "resumed_from_epoch": initial_epoch,
        "epochs_executed": epochs_executed,
        "early_stopped": early_stopped,
        "training_time_seconds": round(training_elapsed, 2),
        "training_time_formatted": _format_duration(training_elapsed),
        "avg_seconds_per_epoch": round(training_elapsed / max(1, n_new), 2),
        "data_counts": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test_used_in_training": 0,
        },
        "config": {
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
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_factor": REDUCE_LR_FACTOR,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
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
        "best_model_verified_val_loss": round(float(verified_val_loss), 6),
        "best_model_verified_val_accuracy": round(float(verified_val_acc), 6),
        "artifacts": {
            "best_model": best_model_path.relative_to(MODELS_DIR.parent).as_posix(),
            "last_model": last_model_path.relative_to(MODELS_DIR.parent).as_posix(),
            "history_csv": TRAINING_HISTORY_CSV.relative_to(MODELS_DIR.parent).as_posix(),
            "history_json": TRAINING_HISTORY_JSON.relative_to(MODELS_DIR.parent).as_posix(),
            "training_report": TRAINING_REPORT_MD.relative_to(MODELS_DIR.parent).as_posix(),
            "accuracy_figure": acc_fig_path.relative_to(MODELS_DIR.parent).as_posix(),
            "loss_figure": loss_fig_path.relative_to(MODELS_DIR.parent).as_posix(),
        },
        "history": history_df.to_dict(orient="list"),
    }

    TRAINING_HISTORY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_training_report(summary=summary, history_df=history_df, report_path=TRAINING_REPORT_MD)

    print("\n================ RINGKASAN HASIL TRAINING ================")
    print(f"Epoch yang dijalankan   : {epochs_executed} / {max_epochs} (best epoch = {best_epoch}, early_stopped = {early_stopped})")
    print(f"Total waktu training    : {summary['training_time_formatted']} ({summary['avg_seconds_per_epoch']:.2f} s/epoch)")
    print(f"Best Epoch #{best_epoch:02d} Metrics :")
    print(f"  Training Accuracy     : {best_row['accuracy'] * 100:.2f}% ({best_row['accuracy']:.4f})")
    print(f"  Validation Accuracy   : {best_row['val_accuracy'] * 100:.2f}% ({best_row['val_accuracy']:.4f})")
    print(f"  Training Loss         : {best_row['loss']:.4f}")
    print(f"  Validation Loss       : {best_row['val_loss']:.4f}")
    print(f"Final Epoch #{int(final_row['epoch']):02d} Metrics:")
    print(f"  Training Accuracy     : {final_row['accuracy'] * 100:.2f}% ({final_row['accuracy']:.4f})")
    print(f"  Validation Accuracy   : {final_row['val_accuracy'] * 100:.2f}% ({final_row['val_accuracy']:.4f})")
    print(f"  Training Loss         : {final_row['loss']:.4f}")
    print(f"  Validation Loss       : {final_row['val_loss']:.4f}")
    print(f"Verifikasi Model Terbaik: val_loss={verified_val_loss:.4f}, val_accuracy={verified_val_acc * 100:.2f}%")
    print(f"Model terbaik tersimpan : {best_model_path}")
    print(f"History tersimpan       : {TRAINING_HISTORY_CSV} & {TRAINING_HISTORY_JSON}")
    print(f"Grafik tersimpan        : {acc_fig_path} & {loss_fig_path}")
    print("==========================================================")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FER-2013 Custom CNN (Train & Validation only).")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Max epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced", "smoothed", "none"],
        default="balanced",
        help="Training class weighting strategy (default: balanced)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--resume", action="store_true", help="Resume training from last saved checkpoint")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        class_weight_mode=args.class_weight_mode,
        random_seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
