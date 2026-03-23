#!/usr/bin/env python3
"""Training entrypoint for the HAR classification pipeline.

Usage
-----
    python scripts/train.py [--dataset uci_har] [--model random_forest] \
        [--data-dir data] [--results-dir results] [--seed 42] \
        [--n-estimators 100] [--fast]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_har.config import build_config_from_args, select_dataset
from ai_har.data import download_and_extract, load_dataset
from ai_har.evaluate import evaluate_model, print_confusion_matrix, save_results
from ai_har.model import build_model, save_model, train


def main(argv=None) -> None:
    cfg = build_config_from_args(argv)

    # Dataset selection: resolve "Human Activity" task → UCI HAR
    chosen = select_dataset(task="Human Activity")
    if cfg.dataset == chosen:
        info = cfg.dataset_info()
        print(f"[config] Selected dataset: {info['name']}")
        print(f"         {info['description']}")

    # 1. Download / cache dataset
    download_and_extract(cfg)

    # 2. Load splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(cfg)

    # 3. Build and train model
    model = build_model(cfg)
    model = train(model, X_train, y_train)

    # 4. Evaluate on validation set
    val_metrics = evaluate_model(model, X_val, y_val, split_name="val")
    save_results(val_metrics, cfg, split_name="val")

    # 5. Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, split_name="test")
    print_confusion_matrix(test_metrics)
    save_results(test_metrics, cfg, split_name="test")

    # 6. Persist trained model
    model_path = cfg.results_dir / f"{cfg.model_type}_model.pkl"
    save_model(model, model_path)

    print("\n[train] Done. Results saved to:", cfg.results_dir)


if __name__ == "__main__":
    main()
