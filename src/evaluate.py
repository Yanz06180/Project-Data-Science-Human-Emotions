"""Model Evaluation Pipeline for FER-2013 Custom CNN on Untouched Test Dataset.

Mengevaluasi model terbaik tersimpan (``models/fer2013_cnn.keras``) terhadap
seluruh dataset pengujian (``data/raw/test/``, 7.178 citra) menggunakan
preprocessing identik dengan fase training (grayscale 1-kanal, 48x48,
normalisasi tunggal [0, 1], tanpa augmentasi).

Jaminan Metodologis:
- Dataset testing HANYA digunakan untuk evaluasi akhir (read-only).
- Tidak ada training ulang, tidak ada tuning hyperparameter dari hasil testing,
  dan tidak ada data augmentation saat evaluasi (``training=False``).
"""

from __future__ import annotations

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
    FIGURES_DIR,
    ID_TO_LABEL,
    INPUT_SHAPE,
    LABEL_TO_ID,
    NUM_CLASSES,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_TEST_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.data_loader import FER2013DataLoader, verify_label_mapping_matches_config
from src.data_splitting import collect_directory_manifest, load_label_mapping, load_split

CLASS_DISPLAY_NAMES: list[str] = [ID_TO_LABEL[i].capitalize() for i in range(NUM_CLASSES)]

EVAL_METRICS_JSON: Path = REPORTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_TXT: Path = REPORTS_DIR / "classification_report.txt"
EVALUATION_REPORT_MD: Path = REPORTS_DIR / "evaluation_report.md"


def load_test_data(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Memuat seluruh citra dari data/raw/test/ menggunakan preprocessing identik tanpa augmentasi."""
    label_to_id, _ = load_label_mapping(SPLITS_DIR)
    verify_label_mapping_matches_config(label_to_id)

    # Verifikasi silang manifest test.csv terhadap seluruh isi aktual data/raw/test/
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

    # Pastikan seluruh path berasal dari data/raw/test/ dan tidak ada augmentasi
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


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Menghitung confusion matrix (baris = True Label, kolom = Predicted Label)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_DISPLAY_NAMES,
) -> dict[str, Any]:
    """Menghitung Accuracy, per-class Precision/Recall/F1/Support, Macro Avg, dan Weighted Avg."""
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class: list[dict[str, Any]] = []
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        pred_count = int(cm[:, idx].sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class.append({
            "label_id": idx,
            "class_name": name,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "support": support,
            "predicted_count": pred_count,
            "correct_count": tp,
        })

    sup_arr = np.asarray(supports, dtype=np.float64)
    prec_arr = np.asarray(precisions, dtype=np.float64)
    rec_arr = np.asarray(recalls, dtype=np.float64)
    f1_arr = np.asarray(f1s, dtype=np.float64)

    macro_avg = {
        "precision": round(float(prec_arr.mean()), 6),
        "recall": round(float(rec_arr.mean()), 6),
        "f1_score": round(float(f1_arr.mean()), 6),
        "support": total,
    }
    weighted_avg = {
        "precision": round(float(np.sum(prec_arr * sup_arr) / total), 6),
        "recall": round(float(np.sum(rec_arr * sup_arr) / total), 6),
        "f1_score": round(float(np.sum(f1_arr * sup_arr) / total), 6),
        "support": total,
    }

    # Identifikasi pasangan kelas yang paling sering tertukar (off-diagonal confusion)
    confused_pairs: list[dict[str, Any]] = []
    for t_idx in range(num_classes):
        row_total = int(cm[t_idx].sum())
        for p_idx in range(num_classes):
            if t_idx == p_idx:
                continue
            cnt = int(cm[t_idx, p_idx])
            if cnt > 0:
                confused_pairs.append({
                    "true_id": t_idx,
                    "true_class": class_names[t_idx],
                    "pred_id": p_idx,
                    "pred_class": class_names[p_idx],
                    "count": cnt,
                    "pct_of_true_class": round(cnt / row_total * 100, 2) if row_total > 0 else 0.0,
                })
    confused_pairs.sort(key=lambda d: d["count"], reverse=True)

    return {
        "accuracy": round(accuracy, 6),
        "total_samples": total,
        "correct_predictions": int(np.trace(cm)),
        "incorrect_predictions": int(total - np.trace(cm)),
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": confused_pairs[:10],
    }


def format_classification_report(metrics: dict[str, Any]) -> str:
    """Memformat Classification Report standar menyerupai scikit-learn."""
    lines = [
        "Classification Report — FER-2013 Custom CNN (Test Set)",
        "=" * 68,
        f"{'Class':<16}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Support':>12}",
        "-" * 68,
    ]
    for row in metrics["per_class"]:
        label_str = f"{row['label_id']} = {row['class_name']}"
        lines.append(
            f"{label_str:<16}{row['precision']:>12.4f}{row['recall']:>12.4f}"
            f"{row['f1_score']:>12.4f}{row['support']:>12d}"
        )
    lines.append("-" * 68)
    lines.append(
        f"{'accuracy':<16}{'':>12}{'':>12}{metrics['accuracy']:>12.4f}{metrics['total_samples']:>12d}"
    )
    m = metrics["macro_avg"]
    w = metrics["weighted_avg"]
    lines.append(
        f"{'macro avg':<16}{m['precision']:>12.4f}{m['recall']:>12.4f}"
        f"{m['f1_score']:>12.4f}{m['support']:>12d}"
    )
    lines.append(
        f"{'weighted avg':<16}{w['precision']:>12.4f}{w['recall']:>12.4f}"
        f"{w['f1_score']:>12.4f}{w['support']:>12d}"
    )
    lines.append("=" * 68)
    return "\n".join(lines)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list[str] = CLASS_DISPLAY_NAMES,
    output_path: Path = FIGURES_DIR / "10_test_confusion_matrix.png",
) -> Path:
    """Membuat heatmap Confusion Matrix dengan nama kelas dan jumlah prediksi (+ %) pada tiap sel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.8), gridspec_kw={"wspace": 0.45})

    # Panel 1: Jumlah Prediksi Absolut (Counts)
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

    # Panel 2: Proporsi per Kelas Aktual (Row-Normalized Recall %)
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

    fig.suptitle("Evaluasi Confusion Matrix — FER-2013 Custom CNN (7.178 Citra Test)", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.15, wspace=0.45)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_class_metrics(
    metrics: dict[str, Any],
    output_path: Path = FIGURES_DIR / "11_test_per_class_metrics.png",
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
    ax.set_ylim(0, min(105, max(df["precision"].max(), df["recall"].max(), df["f1_score"].max()) * 100 + 16))
    ax.set_title("Precision, Recall, dan F1-Score per Kelas Ekspresi — FER-2013 Test Set", fontsize=13.5, fontweight="bold", pad=12)
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
) -> Path:
    """Visualisasi tambahan: Top pasangan kelas tertukar & contoh sampel salah klasifikasi."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_pairs = metrics["top_confused_pairs"][:8]

    fig = plt.figure(figsize=(14.5, 6.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.4, 0.25, 1, 1, 1, 1], hspace=0.42, wspace=0.22)

    # Kiri: Bar chart horizontal 8 pasangan ekspresi paling sering tertukar
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

    # Kanan: 8 contoh citra salah klasifikasi (1 sampel confidence tinggi dari masing-masing top pair)
    for idx_pair, pair in enumerate(top_pairs[:8]):
        r = idx_pair // 4
        c = 2 + (idx_pair % 4)
        ax_img = fig.add_subplot(gs[r, c])

        mask = (y_true == pair["true_id"]) & (y_pred == pair["pred_id"])
        candidate_idxs = np.where(mask)[0]
        if len(candidate_idxs) > 0:
            # Pilih contoh dengan probabilitas prediksi tertinggi (kesalahan paling "yakin")
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

    fig.suptitle("Analisis Kesalahan Klasifikasi — FER-2013 Test Set", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_evaluation_report_md(
    payload: dict[str, Any],
    report_text: str,
    output_path: Path = EVALUATION_REPORT_MD,
) -> Path:
    """Menulis laporan evaluasi Markdown lengkap yang 100% berbasis hasil prediksi aktual."""
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

    md = f"""# Laporan Model Evaluation — FER-2013 Custom CNN (Test Set)

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Tahap:** Tahap 6 — Model Evaluation (Evaluasi Akhir pada Data Testing)  
**Model Dievaluasi:** `{payload['model_path']}`  
**Data Testing:** `data/raw/test/` (`{m['total_samples']:,}` citra; tanpa augmentasi, tanpa training ulang, tanpa tuning)  
**Waktu Evaluasi (UTC):** `{payload['evaluated_at_utc']}`

---

## 1. Ringkasan Metrik Utama (Test Set)

| Metrik | Nilai Aktual |
|---|---:|
| **Test Accuracy** | **{m['accuracy'] * 100:.2f}%** (`{m['correct_predictions']:,}` / `{m['total_samples']:,}` benar) |
| **Test Loss (Sparse Categorical Crossentropy)** | **`{payload['test_loss']:.4f}`** |
| **Top-2 Accuracy** | **{payload['top2_accuracy'] * 100:.2f}%** |
| **Macro Avg Precision / Recall / F1-Score** | `{m['macro_avg']['precision'] * 100:.2f}%` / `{m['macro_avg']['recall'] * 100:.2f}%` / **`{m['macro_avg']['f1_score'] * 100:.2f}%`** |
| **Weighted Avg Precision / Recall / F1-Score** | `{m['weighted_avg']['precision'] * 100:.2f}%` / `{m['weighted_avg']['recall'] * 100:.2f}%` / **`{m['weighted_avg']['f1_score'] * 100:.2f}%`** |
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

## 5. Analisis Hasil Evaluasi

- **Performa Keseluruhan:** Model Custom CNN (`110.983` parameter) mencapai **Test Accuracy {m['accuracy'] * 100:.2f}%**, **Macro F1-Score {m['macro_avg']['f1_score'] * 100:.2f}%**, dan **Weighted F1-Score {m['weighted_avg']['f1_score'] * 100:.2f}%** (dengan Top-2 Accuracy **{payload['top2_accuracy'] * 100:.2f}%**), jauh melampaui baseline tebakan kelas mayoritas (`24,71%` accuracy / `~5,66%` Macro F1 pada test).
- **Kelas dengan Performa Terbaik:** **`{best_f1['class_name']}`** meraih F1-Score tertinggi sebesar **{best_f1['f1_score'] * 100:.2f}%** (Precision `{best_f1['precision'] * 100:.2f}%`, Recall `{best_f1['recall'] * 100:.2f}%`, support `{best_f1['support']:,}`); Recall tertinggi diraih oleh **`{best_rec['class_name']}`** (`{best_rec['recall'] * 100:.2f}%`).
- **Kelas dengan Performa Terendah:** **`{worst_f1['class_name']}`** mencatat F1-Score terendah sebesar **{worst_f1['f1_score'] * 100:.2f}%** (Precision `{worst_f1['precision'] * 100:.2f}%`, Recall `{worst_f1['recall'] * 100:.2f}%`, support `{worst_f1['support']:,}`); Recall terendah berada pada **`{worst_rec['class_name']}`** (`{worst_rec['recall'] * 100:.2f}%`).
- **Ekspresi Paling Sering Tertukar:** Kekeliruan terbanyak secara absolut terjadi pada **{top5[0]['true_class']} → {top5[0]['pred_class']}** (`{top5[0]['count']:,}` sampel; `{top5[0]['pct_of_true_class']:.2f}%` dari `{top5[0]['true_class']}`), diikuti **{top5[1]['true_class']} → {top5[1]['pred_class']}** (`{top5[1]['count']:,}` sampel) dan **{top5[2]['true_class']} → {top5[2]['pred_class']}** (`{top5[2]['count']:,}` sampel).
- **Kemungkinan Penyebab Kesalahan Klasifikasi:**
  1. **Kemiripan fitur mikro wajah antar emosi negatif/berdekatan** pada resolusi rendah `48×48` grayscale (misalnya ketegangan alis/mulut antara `Fear`, `Sad`, `Angry`, `Neutral`, serta mata/mulut terbuka antara `Fear` dan `Surprise`).
  2. **Ketidakseimbangan kelas ekstrem & efek `class_weight`:** Kelas minoritas `Disgust` (hanya `349` citra latih / `1,52%`) diberi bobot tinggi (`9,40×`) saat training sehingga model lebih sensitif memprediksi `Disgust` (menaikkan recall namun dapat menekan precision dan menggeser sebagian prediksi `Angry`/`Fear`/`Sad`).
  3. **Noise inheren dataset FER-2013** (yang ditemukan saat inspeksi & EDA): ambiguitas pelabelan manusia, duplikat lintas kelas, oklusi wajah oleh tangan, citra non-foto/watermark, serta resolusi kecil `48×48`.
  4. **Kapasitas arsitektur ringkas (`111K` parameter):** 3 blok konvolusi dengan satu `Conv2D` per blok membatasi kemampuan mengekstrak nuansa ekspresi wajah yang sangat halus dibanding blok konvolusi ganda/lebih dalam.

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
    model_path: Path = BEST_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Menjalankan evaluasi penuh model terbaik terhadap seluruh data/raw/test/."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model terbaik tidak ditemukan di {model_path}. Pastikan src/train.py sudah selesai."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== MEMUAT MODEL TERBAIK & DATASET TESTING (READ-ONLY) ===", flush=True)
    t0 = time.perf_counter()
    model = keras.models.load_model(model_path)
    test_df, x_test, y_test = load_test_data(batch_size=batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Evaluasi resmi (training=False otomatis di evaluate() dan predict())
    test_loss, keras_test_acc = model.evaluate(test_ds, verbose=0)
    y_probs = model.predict(test_ds, verbose=0)
    eval_elapsed = time.perf_counter() - t0

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

    report_text = format_classification_report(metrics)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    # Simpan seluruh visualisasi & laporan
    cm_fig = plot_confusion_matrix_heatmap(cm, class_names=CLASS_DISPLAY_NAMES)
    metrics_fig = plot_per_class_metrics(metrics)
    err_fig = plot_misclassification_analysis(x_test, y_test, y_pred, y_probs, metrics)

    CLASSIFICATION_REPORT_TXT.write_text(report_text + "\n", encoding="utf-8")

    payload: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "test_source": "data/raw/test/",
        "evaluation_time_seconds": round(eval_elapsed, 2),
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
            "classification_report_txt": CLASSIFICATION_REPORT_TXT.relative_to(PROJECT_ROOT).as_posix(),
            "metrics_json": EVAL_METRICS_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_report_md": EVALUATION_REPORT_MD.relative_to(PROJECT_ROOT).as_posix(),
        },
    }

    EVAL_METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_evaluation_report_md(payload, report_text, output_path=EVALUATION_REPORT_MD)

    print(f"\nEvaluasi selesai dalam {eval_elapsed:.1f}s pada {len(y_test):,} citra test.")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {metrics['accuracy'] * 100:.2f}% ({metrics['correct_predictions']:,}/{metrics['total_samples']:,})")
    print(f"Top-2 Accuracy: {top2_acc * 100:.2f}%")
    print(f"Macro F1      : {metrics['macro_avg']['f1_score'] * 100:.2f}%")
    print(f"Weighted F1   : {metrics['weighted_avg']['f1_score'] * 100:.2f}%\n")
    print(report_text)
    print("\nTop 5 Pasangan Ekspresi Tertukar:")
    for p in metrics["top_confused_pairs"][:5]:
        print(f"  {p['true_class']} -> {p['pred_class']}: {p['count']} ({p['pct_of_true_class']:.2f}%)")
    print(f"\nGrafik tersimpan : {cm_fig}, {metrics_fig}, {err_fig}")
    print(f"Laporan tersimpan: {CLASSIFICATION_REPORT_TXT}, {EVAL_METRICS_JSON}, {EVALUATION_REPORT_MD}")
    return payload


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
