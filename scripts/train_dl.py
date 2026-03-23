#!/usr/bin/env python3
"""Deep-learning training entrypoint for HAR using a 1D CNN on raw signals.

Loads the raw inertial-signal sequences (9 channels × 128 timesteps) from the
UCI HAR Dataset, trains a lightweight PyTorch 1D CNN, and saves the model
weights and evaluation metrics to disk.

Usage
-----
    # Quick demo (≈ 500 training samples, 3 epochs — runs in seconds on CPU):
    python scripts/train_dl.py --fast

    # Standard run (full dataset, 10 epochs):
    python scripts/train_dl.py

    # Custom hyper-parameters:
    python scripts/train_dl.py --epochs 5 --batch-size 64 --lr 1e-3

    # Custom paths:
    python scripts/train_dl.py --data-dir /mnt/datasets --results-dir /mnt/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_har.config import Config, select_dataset
from ai_har.data import download_and_extract, load_inertial_signals
from ai_har.dl_model import (
    CNN1D,
    DLConfig,
    evaluate_dl,
    save_dl_model,
    save_dl_results,
    train_dl,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> DLConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 1D CNN on UCI HAR raw inertial signals (deep-learning path)"
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        dest="random_seed",
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for downloaded datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for saving results (default: %(default)s)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        dest="fast_mode",
        help=(
            "Fast/demo mode: subsample to 500 training samples and run only 3 epochs"
        ),
    )
    args = parser.parse_args(argv)
    return DLConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        random_seed=args.random_seed,
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        fast_mode=args.fast_mode,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    dl_cfg = _parse_args(argv)

    # Reproducibility
    torch.manual_seed(dl_cfg.random_seed)
    np.random.seed(dl_cfg.random_seed)

    # Dataset selection
    select_dataset(task="Human Activity")

    # 1. Download dataset if needed (reuse existing ML config for paths)
    ml_cfg = Config(data_dir=dl_cfg.data_dir, dataset="uci_har")
    download_and_extract(ml_cfg)

    # 2. Load raw inertial-signal sequences — shape (N, 9, 128)
    print("[dl] Loading raw inertial signals …")
    X_train_full, y_train_full = load_inertial_signals(ml_cfg, "train")
    X_test, y_test = load_inertial_signals(ml_cfg, "test")

    # 3. Optional subsampling for fast/demo mode
    if dl_cfg.fast_mode:
        rng = np.random.default_rng(dl_cfg.random_seed)
        n = min(dl_cfg.fast_n_samples, len(X_train_full))
        idx = rng.choice(len(X_train_full), size=n, replace=False)
        X_train_full = X_train_full[idx]
        y_train_full = y_train_full[idx]
        print(
            f"[dl] Fast mode: {len(X_train_full)} training samples, "
            f"{dl_cfg.fast_epochs} epochs"
        )

    # 4. Validation split (10 % of training data)
    rng = np.random.default_rng(dl_cfg.random_seed)
    n_val = max(1, int(len(X_train_full) * 0.1))
    val_idx = rng.choice(len(X_train_full), size=n_val, replace=False)
    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_idx] = False

    X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    n_channels = X_train.shape[1]  # 9
    print(
        f"[dl] Splits — train: {X_train.shape}, "
        f"val: {X_val.shape}, test: {X_test.shape}"
    )

    # 5. Build and train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(n_channels=n_channels, n_classes=6)
    model = train_dl(model, X_train, y_train, X_val, y_val, dl_cfg, device=device)

    # 6. Evaluate on validation and test sets
    val_metrics = evaluate_dl(model, X_val, y_val, split_name="val", device=device)
    save_dl_results(val_metrics, dl_cfg.results_dir, split_name="val")

    test_metrics = evaluate_dl(model, X_test, y_test, split_name="test", device=device)
    save_dl_results(test_metrics, dl_cfg.results_dir, split_name="test")

    # 7. Persist model weights
    model_path = dl_cfg.results_dir / "cnn1d_model.pt"
    save_dl_model(model, model_path)

    print("\n[dl] Done. Results saved to:", dl_cfg.results_dir)


if __name__ == "__main__":
    main()
