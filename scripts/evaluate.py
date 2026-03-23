#!/usr/bin/env python3
"""Evaluation-only entrypoint for the HAR classification pipeline.

Loads a previously trained model and re-evaluates it on the test set.

Usage
-----
    python scripts/evaluate.py [--dataset uci_har] [--model random_forest] \
        [--data-dir data] [--results-dir results]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_har.config import build_config_from_args
from ai_har.data import load_dataset
from ai_har.evaluate import evaluate_model, print_confusion_matrix, save_results
from ai_har.model import load_model


def main(argv=None) -> None:
    cfg = build_config_from_args(argv)

    model_path = cfg.results_dir / f"{cfg.model_type}_model.pkl"
    if not model_path.exists():
        print(
            f"[evaluate] No saved model found at '{model_path}'. "
            "Run scripts/train.py first."
        )
        sys.exit(1)

    model = load_model(model_path)

    _, _, _, _, X_test, y_test = load_dataset(cfg)

    metrics = evaluate_model(model, X_test, y_test, split_name="test")
    print_confusion_matrix(metrics)
    save_results(metrics, cfg, split_name="test_eval")

    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()
