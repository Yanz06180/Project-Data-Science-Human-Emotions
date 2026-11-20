"""Model Evaluation Pipeline for FER-2013 Models on Untouched Test Dataset.

Mengevaluasi model tersimpan terhadap seluruh dataset pengujian
(``data/raw/test/``, 7.178 citra) menggunakan preprocessing identik dengan fase
training (grayscale 1-kanal, 48x48, normalisasi tunggal [0, 1], tanpa augmentasi).

Mendukung CLI argumen (``--experiment``, ``--model-path``, ``--output-prefix``)
serta mencatat metrik deployment PRD (``model_size_mb`` dan
``inference_time_ms_per_image`` dengan 1-batch warm-up) tanpa menimpa artefak
Baseline.

Jaminan Metodologis:
- Dataset testing HANYA digunakan untuk evaluasi akhir (read-only).
- Tidak ada training ulang, tidak ada tuning hyperparameter dari hasil testing,
  dan tidak ada data augmentation saat evaluasi (``training=False``).
"""

from __future__ import annotations

import argparse
import json
import os
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
    EXP2_BEST_MODEL_PATH,
    FIGURES_DIR,
    INPUT_SHAPE,
    MODELS_DIR,
    NUM_CLASSES,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_TEST_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.data_loader import FER2013DataLoader, verify_label_mapping_matches_config
from src.data_splitting import collect_directory_manifest, load_label_mapping, load_split
from src.metrics import (
    CLASS_DISPLAY_NAMES,
    compute_classification_metrics,
    format_classification_report,
)

# Default baseline output paths (preserved for backward compatibility)
EVAL_METRICS_JSON: Path = REPORTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_TXT: Path = REPORTS_DIR / "classification_report.txt"
EVALUATION_REPORT_MD: Path = REPORTS_DIR / "evaluation_report.md"


def configure_gpu() -> str:
    """Enable GPU memory growth safely if available; return human-readable device string."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return f"GPU ({len(gpus)} device)"
    return "CPU (TensorFlow oneDNN)"


def resolve_default_model_path(experiment: str) -> Path:
    """Resolve canonical model file path for an experiment name."""
    if experiment == "baseline":
        return BEST_MODEL_PATH
    if experiment == "exp2":
        return EXP2_BEST_MODEL_PATH
    return MODELS_DIR / f"fer2013_{experiment}.keras"


def resolve_evaluation_paths(
    experiment: str = "baseline",
    output_prefix: str | None = None,
) -> dict[str, Path]:
    """Resolve isolated evaluation report & figure paths so baseline is never overwritten."""
    if output_prefix is not None:
        clean_prefix = output_prefix.strip()
        if clean_prefix and not clean_prefix.endswith("_"):
            clean_prefix = f"{clean_prefix}_"
    else:
        clean_prefix = "" if experiment == "baseline" else f"{experiment}_"

    if clean_prefix == "":
        return {
            "classification_report_txt": CLASSIFICATION_REPORT_TXT,
            "metrics_json": EVAL_METRICS_JSON,
            "evaluation_report_md": EVALUATION_REPORT_MD,
            "confusion_matrix_figure": FIGURES_DIR / "10_test_confusion_matrix.png",
            "per_class_metrics_figure": FIGURES_DIR / "11_test_per_class_metrics.png",
            "misclassification_figure": FIGURES_DIR / "12_test_misclassification_analysis.png",
        }

    return {
        "classification_report_txt": REPORTS_DIR / f"{clean_prefix}classification_report.txt",
        "metrics_json": REPORTS_DIR / f"{clean_prefix}evaluation_metrics.json",
        "evaluation_report_md": REPORTS_DIR / f"{clean_prefix}evaluation_report.md",
        "confusion_matrix_figure": FIGURES_DIR / f"{clean_prefix}test_confusion_matrix.png",
        "per_class_metrics_figure": FIGURES_DIR / f"{clean_prefix}test_per_class_metrics.png",
        "misclassification_figure": FIGURES_DIR / f"{clean_prefix}test_misclassification_analysis.png",
    }


def load_test_data(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Memuat seluruh citra dari data/raw/test/ menggunakan preprocessing identik tanpa augmentasi."""
    label_to_id, _ = load_label_mapping(SPLITS_DIR)
    verify_label_mapping_matches_config(label_to_id)

    raw_test_manifest = collect_directory_manifest(RAW_TEST_DIR, label_to_id, project_root=PROJECT_ROOT)
    test_csv_path = SPLITS_DIR / "test.csv"
    if test_csv_path.exists():
        test_df = load_split("test", splits_dir=SPLITS_DIR, project_root=PROJECT_ROOT, resolve_paths=True)
        assert set(test_df["filepath"]) == set(raw_test_manifest["filepath"]), (
            "Manifest data/splits/test.csv tidak sinkron dengan data/raw/test/!"
        )
    else:
        test_df = raw_test_manifest.copy()
        test_df["split"] = "test"
        test_df["abs_path"] = test_df["filepath"].map(lambda rel: str((PROJECT_ROOT / rel).resolve()))

    assert all(p.startswith("data/raw/test/") for p in test_df["filepath"])
    test_loader = FER2013DataLoader(
        df=test_df,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        random_seed=RANDOM_SEED,
    )
    x_test, y_test = test_loader.load_all_arrays(augment=False)

    assert x_test.shape == (len(test_df), *INPUT_SHAPE), f"Shape test tidak sesuai: {x_test.shape}"
    assert x_test.dtype == np.float32
    assert 0.0 <= float(x_test.min()) <= float(x_test.max()) <= 1.0
    return test_df, x_test, y_test


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list[str] = CLASS_DISPLAY_NAMES,
    output_path: Path = FIGURES_DIR / "10_test_confusion_matrix.png",
    title_label: str = "FER-2013 Custom CNN",
) -> Path:
    """Membuat heatmap Confusion Matrix dengan nama kelas dan jumlah prediksi (+ %) pada tiap sel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.8), gridspec_kw={"wspace": 0.45})

    im1 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Jumlah Prediksi", fontsize=10)
    axes[0].set_title("Confusion Matrix (Jumlah Prediksi) — Test Set", fontsize=12.5, fontweight="bold", pad=12)
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    axes[0].set_yticklabels(class_names, fontsize=10)
    axes[0].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11, fontweight="bold")

    thresh1 = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh1 else "#111111"
            weight = "bold" if i == j else "normal"
            axes[0].text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9.5, fontweight=weight)

    im2 = axes[1].imshow(cm_norm * 100.0, interpolation="nearest", cmap="YlGnBu", vmin=0.0, vmax=100.0)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Persentase per True Class (%)", fontsize=10)
    axes[1].set_title("Confusion Matrix Ternormalisasi (% per Kelas Aktual)", fontsize=12.5, fontweight="bold", pad=12)
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    axes[1].set_yticklabels(class_names, fontsize=10)
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11, fontweight="bold")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100.0
            color = "white" if pct > 52.0 else "#111111"
            weight = "bold" if i == j else "normal"
            axes[1].text(
                j,
                i,
                f"{pct:.1f}%\n({cm[i, j]})",
                ha="center",
                va="center",
                color=color,
                fontsize=8.2,
                fontweight=weight,
            )

    fig.suptitle(f"Evaluasi Confusion Matrix — {title_label} (7.178 Citra Test)", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.15, wspace=0.45)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_class_metrics(
    metrics: dict[str, Any],
    output_path: Path = FIGURES_DIR / "11_test_per_class_metrics.png",
    title_label: str = "FER-2013 Test Set",
) -> Path:
    """Visualisasi Precision, Recall, dan F1-Score per kelas beserta Macro & Weighted F1."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics["per_class"])
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", color="#E0E0E0", linewidth=0.7)

    b1 = ax.bar(x - width, df["precision"] * 100, width, label="Precision", color="#457B9D", edgecolor="#222222", linewidth=0.6)
    b2 = ax.bar(x, df["recall"] * 100, width, label="Recall", color="#2A9D8F", edgecolor="#222222", linewidth=0.6)
    b3 = ax.bar(x + width, df["f1_score"] * 100, width, label="F1-Score", color="#E76F51", edgecolor="#222222", linewidth=0.6)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    macro_f1 = metrics["macro_avg"]["f1_score"] * 100
    weighted_f1 = metrics["weighted_avg"]["f1_score"] * 100
    ax.axhline(macro_f1, color="#8E44AD", linestyle="--", linewidth=1.4, label=f"Macro Avg F1 ({macro_f1:.2f}%)")
    ax.axhline(weighted_f1, color="#D62828", linestyle="-.", linewidth=1.4, label=f"Weighted Avg F1 ({weighted_f1:.2f}%)")

    x_labels = [f"{r['class_name']}\n(n={r['support']:,})" for _, r in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Skor (%)")
    ax.set_ylim(0, min(108, max(df["precision"].max(), df["recall"].max(), df["f1_score"].max()) * 100 + 18))
    ax.set_title(f"Precision, Recall, dan F1-Score per Kelas Ekspresi — {title_label}", fontsize=13.5, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_misclassification_analysis(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    metrics: dict[str, Any],
    output_path: Path = FIGURES_DIR / "12_test_misclassification_analysis.png",
    title_label: str = "FER-2013 Test Set",
) -> Path:
    """Visualisasi tambahan: Top pasangan kelas tertukar & contoh sampel salah klasifikasi."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_pairs = metrics["top_confused_pairs"][:8]

    fig = plt.figure(figsize=(14.5, 6.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.4, 0.25, 1, 1, 1, 1], hspace=0.42, wspace=0.22)

    ax_bar = fig.add_subplot(gs[:, 0])
    ax_bar.set_facecolor("#FAFAFA")
    ax_bar.grid(axis="x", linestyle="--", color="#E0E0E0", linewidth=0.7)

    pair_labels = [f"{p['true_class']} → {p['pred_class']}" for p in reversed(top_pairs)]
    pair_counts = [p["count"] for p in reversed(top_pairs)]
    pair_pcts = [p["pct_of_true_class"] for p in reversed(top_pairs)]
    y_pos = np.arange(len(pair_labels))

    colors = plt.cm.OrRd(np.linspace(0.35, 0.85, len(pair_labels)))
    bars = ax_bar.barh(y_pos, pair_counts, color=colors, edgecolor="#333333", height=0.65)
    for bar, cnt, pct in zip(bars, pair_counts, pair_pcts):
        ax_bar.text(
            cnt + max(pair_counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(pair_labels, fontsize=9.2, fontweight="bold")
    ax_bar.set_xlabel("Jumlah Salah Klasifikasi")
    ax_bar.set_xlim(0, max(pair_counts) * 1.32)
    ax_bar.set_title("Top 8 Pasangan Ekspresi Tertukar\n(True → Predicted)", fontsize=11.5, fontweight="bold", pad=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for idx_pair, pair in enumerate(top_pairs[:8]):
        r = idx_pair // 4
        c = 2 + (idx_pair % 4)
        ax_img = fig.add_subplot(gs[r, c])

        mask = (y_true == pair["true_id"]) & (y_pred == pair["pred_id"])
        candidate_idxs = np.where(mask)[0]
        if len(candidate_idxs) > 0:
            confs = y_probs[candidate_idxs, pair["pred_id"]]
            best_idx = int(candidate_idxs[int(np.argmax(confs))])
            conf_val = float(y_probs[best_idx, pair["pred_id"]]) * 100.0
            ax_img.imshow(x_test[best_idx, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax_img.set_title(
                f"T: {pair['true_class']}\nP: {pair['pred_class']} ({conf_val:.0f}%)",
                fontsize=8.2,
                color="#B00020",
                fontweight="bold",
            )
        ax_img.set_xticks([])
        ax_img.set_yticks([])

    fig.suptitle(f"Analisis Kesalahan Klasifikasi — {title_label}", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_evaluation_report_md(
    payload: dict[str, Any],
    report_text: str,
    output_path: Path,
) -> Path:
    """Menulis laporan evaluasi Markdown lengkap termasuk metrik deployment PRD."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = payload["metrics"]
    per_class = m["per_class"]
    best_f1 = max(per_class, key=lambda r: r["f1_score"])
    worst_f1 = min(per_class, key=lambda r: r["f1_score"])
    best_rec = max(per_class, key=lambda r: r["recall"])
    worst_rec = min(per_class, key=lambda r: r["recall"])
    top5 = m["top_confused_pairs"][:5]

    per_class_rows = "\n".join(
        f"| `{r['label_id']}` | {r['class_name']} | {r['precision'] * 100:.2f}% | "
        f"{r['recall'] * 100:.2f}% | {r['f1_score'] * 100:.2f}% | {r['support']:,} | {r['predicted_count']:,} |"
        for r in per_class
    )
    top_conf_rows = "\n".join(
        f"| {i + 1} | **{p['true_class']} → {p['pred_class']}** | `{p['count']:,}` | {p['pct_of_true_class']:.2f}% dari kelas `{p['true_class']}` |"
        for i, p in enumerate(top5)
    )

    md = f"""# Laporan Model Evaluation — {payload['experiment'].upper()} ({payload['model_name']})

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013)  
**Eksperimen:** `{payload['experiment']}`  
**Model Dievaluasi:** `{payload['model_path']}` (`{payload['model_params']:,}` parameter, `{payload['model_size_mb']:.4f} MB`)  
**Data Testing:** `data/raw/test/` (`{m['total_samples']:,}` citra; tanpa augmentasi, tanpa training ulang, tanpa tuning)  
**Perangkat & Waktu Evaluasi (UTC):** `{payload['device']}` | `{payload['evaluated_at_utc']}`

---

## 1. Ringkasan Metrik Utama & Deployment PRD (Test Set)

| Metrik | Nilai Aktual |
|---|---:|
| **Test Accuracy** | **{m['accuracy'] * 100:.2f}%** (`{m['correct_predictions']:,}` / `{m['total_samples']:,}` benar) |
| **Test Loss (Sparse Categorical Crossentropy)** | **`{payload['test_loss']:.4f}`** |
| **Top-2 Accuracy** | **{payload['top2_accuracy'] * 100:.2f}%** |
| **Macro Avg Precision / Recall / F1-Score** | `{m['macro_avg']['precision'] * 100:.2f}%` / `{m['macro_avg']['recall'] * 100:.2f}%` / **`{m['macro_avg']['f1_score'] * 100:.2f}%`** |
| **Weighted Avg Precision / Recall / F1-Score** | `{m['weighted_avg']['precision'] * 100:.2f}%` / `{m['weighted_avg']['recall'] * 100:.2f}%` / **`{m['weighted_avg']['f1_score'] * 100:.2f}%`** |
| **Model Size (`model_size_mb`)** | **`{payload['model_size_mb']:.4f} MB`** |
| **Inference Time (`inference_time_ms_per_image`)** | **`{payload['inference_time_ms_per_image']:.4f} ms/citra`** (`{payload['total_inference_time_sec']:.3f}s` total, setelah 1-batch warm-up) |
| **Rata-rata Confidence (Benar vs Salah)** | `{payload['confidence_stats']['mean_conf_correct'] * 100:.2f}%` vs `{payload['confidence_stats']['mean_conf_incorrect'] * 100:.2f}%` |

---

## 2. Performa per Kelas Ekspresi (0–6)

| ID | Kelas | Precision | Recall | F1-Score | Support | Jumlah Diprediksi |
|---:|---|---:|---:|---:|---:|---:|
{per_class_rows}

---

## 3. Classification Report Lengkap

```text
{report_text}
```

---

## 4. Pasangan Ekspresi Paling Sering Tertukar

| Peringkat | Pola Kekeliruan (True → Predicted) | Jumlah | Proporsi dari Kelas Aktual |
|---:|---|---:|---:|
{top_conf_rows}

---

## 5. Ringkasan Analisis Evaluasi

- **Performa Keseluruhan:** Model `{payload['model_name']}` mencapai **Test Accuracy {m['accuracy'] * 100:.2f}%**, **Macro F1-Score {m['macro_avg']['f1_score'] * 100:.2f}%**, dan **Weighted F1-Score {m['weighted_avg']['f1_score'] * 100:.2f}%** (Top-2 Accuracy **{payload['top2_accuracy'] * 100:.2f}%**).
- **Kelas Terbaik:** **`{best_f1['class_name']}`** (F1 **{best_f1['f1_score'] * 100:.2f}%**, Precision `{best_f1['precision'] * 100:.2f}%`, Recall `{best_f1['recall'] * 100:.2f}%`); Recall tertinggi pada **`{best_rec['class_name']}`** (`{best_rec['recall'] * 100:.2f}%`).
- **Kelas Terendah:** **`{worst_f1['class_name']}`** (F1 **{worst_f1['f1_score'] * 100:.2f}%**, Precision `{worst_f1['precision'] * 100:.2f}%`, Recall `{worst_f1['recall'] * 100:.2f}%`); Recall terendah pada **`{worst_rec['class_name']}`** (`{worst_rec['recall'] * 100:.2f}%`).
- **Pasangan Tertukar Utama:** **{top5[0]['true_class']} → {top5[0]['pred_class']}** (`{top5[0]['count']:,}`), **{top5[1]['true_class']} → {top5[1]['pred_class']}** (`{top5[1]['count']:,}`), dan **{top5[2]['true_class']} → {top5[2]['pred_class']}** (`{top5[2]['count']:,}`).

---

## 6. Artefak Evaluasi Tersimpan

- Confusion Matrix Heatmap: `{payload['artifacts']['confusion_matrix_figure']}`
- Grafik Metrik per Kelas: `{payload['artifacts']['per_class_metrics_figure']}`
- Analisis Kesalahan Klasifikasi: `{payload['artifacts']['misclassification_figure']}`
- Classification Report (TXT): `{payload['artifacts']['classification_report_txt']}`
- Ringkasan Metrik (JSON): `{payload['artifacts']['metrics_json']}`
"""
    output_path.write_text(md, encoding="utf-8")
    return output_path


def evaluate(
    model_path: Path | None = None,
    experiment: str = "baseline",
    output_prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Menjalankan evaluasi penuh model terhadap seluruh data/raw/test/."""
    device_str = configure_gpu()
    resolved_model_path = model_path if model_path is not None else resolve_default_model_path(experiment)
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {resolved_model_path}. Pastikan training sudah selesai."
        )

    out_paths = resolve_evaluation_paths(experiment=experiment, output_prefix=output_prefix)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== EVALUASI '{experiment}' ({resolved_model_path.name}) PADA TEST SET ===", flush=True)
    t_total_start = time.perf_counter()
    model = keras.models.load_model(resolved_model_path)
    model_size_mb = round(os.path.getsize(resolved_model_path) / (1024 ** 2), 4)

    _, x_test, y_test = load_test_data(batch_size=batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_loss, keras_test_acc = model.evaluate(test_ds, verbose=0)

    # 1-batch warm-up sebelum timer inferensi agar tidak bias inisialisasi graf TensorFlow
    _ = model.predict(x_test[:batch_size], batch_size=batch_size, verbose=0)
    t_inf_start = time.perf_counter()
    y_probs = model.predict(test_ds, verbose=0)
    total_inference_time_sec = time.perf_counter() - t_inf_start
    inference_time_ms_per_image = (total_inference_time_sec / len(y_test)) * 1000.0
    eval_elapsed = time.perf_counter() - t_total_start

    assert y_probs.shape == (len(y_test), NUM_CLASSES), f"Output probabilitas salah: {y_probs.shape}"
    assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-4), "Probabilitas softmax tidak berjumlah 1!"

    y_pred = np.argmax(y_probs, axis=1).astype(np.int64)
    top2_preds = np.argsort(y_probs, axis=1)[:, -2:]
    top2_acc = float(np.mean([y_test[i] in top2_preds[i] for i in range(len(y_test))]))

    max_probs = np.max(y_probs, axis=1)
    correct_mask = y_pred == y_test
    conf_stats = {
        "mean_conf_all": round(float(max_probs.mean()), 6),
        "mean_conf_correct": round(float(max_probs[correct_mask].mean()), 6) if correct_mask.any() else 0.0,
        "mean_conf_incorrect": round(float(max_probs[~correct_mask].mean()), 6) if (~correct_mask).any() else 0.0,
    }

    metrics = compute_classification_metrics(y_test, y_pred, class_names=CLASS_DISPLAY_NAMES)
    assert abs(metrics["accuracy"] - float(keras_test_acc)) < 1e-4, (
        f"Selisih akurasi Keras ({keras_test_acc:.6f}) vs manual ({metrics['accuracy']:.6f})!"
    )

    title_label = f"FER-2013 {experiment.upper()} ({model.name})"
    report_text = format_classification_report(
        metrics,
        title=f"Classification Report — {title_label} (Test Set)",
    )
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    cm_fig = plot_confusion_matrix_heatmap(
        cm,
        class_names=CLASS_DISPLAY_NAMES,
        output_path=out_paths["confusion_matrix_figure"],
        title_label=title_label,
    )
    metrics_fig = plot_per_class_metrics(
        metrics,
        output_path=out_paths["per_class_metrics_figure"],
        title_label=f"{title_label} (Test Set)",
    )
    err_fig = plot_misclassification_analysis(
        x_test,
        y_test,
        y_pred,
        y_probs,
        metrics,
        output_path=out_paths["misclassification_figure"],
        title_label=f"{title_label} (Test Set)",
    )

    out_paths["classification_report_txt"].write_text(report_text + "\n", encoding="utf-8")

    payload: dict[str, Any] = {
        "experiment": experiment,
        "model_name": model.name,
        "model_params": int(model.count_params()),
        "model_size_mb": model_size_mb,
        "device": device_str,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": resolved_model_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        "test_source": "data/raw/test/",
        "evaluation_time_seconds": round(eval_elapsed, 2),
        "total_inference_time_sec": round(total_inference_time_sec, 4),
        "inference_time_ms_per_image": round(inference_time_ms_per_image, 4),
        "test_loss": round(float(test_loss), 6),
        "keras_test_accuracy": round(float(keras_test_acc), 6),
        "top2_accuracy": round(top2_acc, 6),
        "confidence_stats": conf_stats,
        "label_mapping": {str(i): CLASS_DISPLAY_NAMES[i] for i in range(NUM_CLASSES)},
        "metrics": metrics,
        "artifacts": {
            "confusion_matrix_figure": cm_fig.relative_to(PROJECT_ROOT).as_posix(),
            "per_class_metrics_figure": metrics_fig.relative_to(PROJECT_ROOT).as_posix(),
            "misclassification_figure": err_fig.relative_to(PROJECT_ROOT).as_posix(),
            "classification_report_txt": out_paths["classification_report_txt"].relative_to(PROJECT_ROOT).as_posix(),
            "metrics_json": out_paths["metrics_json"].relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_report_md": out_paths["evaluation_report_md"].relative_to(PROJECT_ROOT).as_posix(),
        },
    }

    out_paths["metrics_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_evaluation_report_md(payload, report_text, output_path=out_paths["evaluation_report_md"])

    print(f"\nEvaluasi selesai dalam {eval_elapsed:.1f}s pada {len(y_test):,} citra test.")
    print(f"Model Size       : {model_size_mb:.4f} MB ({model.count_params():,} params)")
    print(f"Inference Time   : {inference_time_ms_per_image:.4f} ms/citra ({total_inference_time_sec:.3f}s total)")
    print(f"Test Loss        : {test_loss:.4f}")
    print(f"Test Accuracy    : {metrics['accuracy'] * 100:.2f}% ({metrics['correct_predictions']:,}/{metrics['total_samples']:,})")
    print(f"Top-2 Accuracy   : {top2_acc * 100:.2f}%")
    print(f"Macro F1         : {metrics['macro_avg']['f1_score'] * 100:.2f}%")
    print(f"Weighted F1      : {metrics['weighted_avg']['f1_score'] * 100:.2f}%\n")
    print(report_text)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FER-2013 Keras model on the test set.")
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment identifier (default: baseline)")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional explicit path to .keras model")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output filename prefix")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate(
        model_path=args.model_path,
        experiment=args.experiment,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
