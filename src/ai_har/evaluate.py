"""Evaluation utilities: metrics, confusion matrix, results persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from ai_har.config import Config
from ai_har.data import ACTIVITY_LABELS


def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str = "test",
) -> Dict:
    """Run predictions and compute evaluation metrics.

    Returns
    -------
    dict
        Contains accuracy, per-class precision/recall/F1, and a flat list
        representation of the confusion matrix.
    """
    y_pred = model.predict(X)
    class_names: List[str] = list(ACTIVITY_LABELS.values())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n[evaluate] {split_name.upper()} accuracy: {acc:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    return {
        "split": split_name,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def save_results(metrics: Dict, cfg: Config, split_name: str = "test") -> Path:
    """Persist *metrics* dict as JSON inside :attr:`Config.results_dir`."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_dir / f"{cfg.model_type}_{split_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Results saved to '{out_path}'")
    return out_path


def print_confusion_matrix(metrics: Dict) -> None:
    """Pretty-print the confusion matrix stored in *metrics*."""
    class_names = metrics["class_names"]
    cm = np.array(metrics["confusion_matrix"])
    header = "{:>25s}".format("") + "".join(
        f"{name:>22s}" for name in class_names
    )
    print("\nConfusion matrix:")
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>25s}" + "".join(f"{v:>22d}" for v in row)
        print(row_str)
