"""Shared classification metrics and reporting utilities for FER-2013 models.

Decouples metric calculation from ``src/evaluate.py`` so both ``src/train.py``
(validation diagnostics) and ``src/evaluate.py`` (final test evaluation) can
import identical, verified metric functions without cross-module coupling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import ID_TO_LABEL, NUM_CLASSES

CLASS_DISPLAY_NAMES: list[str] = [ID_TO_LABEL[i].capitalize() for i in range(NUM_CLASSES)]


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute confusion matrix (rows = True Label, cols = Predicted Label)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_DISPLAY_NAMES,
) -> dict[str, Any]:
    """Compute Accuracy, per-class Precision/Recall/F1/Support, Macro Avg, and Weighted Avg."""
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

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
        "precision": round(float(np.sum(prec_arr * sup_arr) / total), 6) if total > 0 else 0.0,
        "recall": round(float(np.sum(rec_arr * sup_arr) / total), 6) if total > 0 else 0.0,
        "f1_score": round(float(np.sum(f1_arr * sup_arr) / total), 6) if total > 0 else 0.0,
        "support": total,
    }

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


def format_classification_report(
    metrics: dict[str, Any],
    title: str = "Classification Report — FER-2013 (Test Set)",
) -> str:
    """Format a scikit-learn-style text classification report."""
    lines = [
        title,
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
