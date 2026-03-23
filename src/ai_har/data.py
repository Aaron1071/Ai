"""Dataset download, extraction, and loading for the UCI HAR Dataset."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Tuple
from urllib.request import urlopen, Request

import numpy as np

from ai_har.config import Config

# Activity label mapping (1-indexed in the raw data)
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _dataset_root(cfg: Config) -> Path:
    """Return the root directory of the extracted UCI HAR Dataset."""
    info = cfg.dataset_info()
    return cfg.data_dir / info["extract_subdir"]


def download_and_extract(cfg: Config, force: bool = False) -> Path:
    """Download the UCI HAR Dataset zip and extract it if not already present.

    Parameters
    ----------
    cfg:
        Project configuration object.
    force:
        Re-download even if the dataset directory already exists.

    Returns
    -------
    Path
        Path to the extracted dataset root directory.
    """
    root = _dataset_root(cfg)
    if root.exists() and not force:
        print(f"[data] Dataset already present at '{root}'. Skipping download.")
        return root

    info = cfg.dataset_info()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cfg.data_dir / info["zip_name"]

    if not zip_path.exists() or force:
        url = info["url"]
        print(f"[data] Downloading {info['name']} from:\n  {url}")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as response:  # noqa: S310
            data = response.read()
        zip_path.write_bytes(data)
        print(f"[data] Saved zip to '{zip_path}' ({len(data) // 1024} KB)")
    else:
        print(f"[data] Zip already present at '{zip_path}'. Extracting …")

    print(f"[data] Extracting to '{cfg.data_dir}' …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cfg.data_dir)

    print(f"[data] Extraction complete → '{root}'")
    return root


def _load_split(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X (features) and y (labels) for *split* ('train' or 'test')."""
    split_dir = root / split
    X = np.loadtxt(split_dir / "X_{}.txt".format(split))
    y = np.loadtxt(split_dir / "y_{}.txt".format(split), dtype=int)
    return X, y


def load_dataset(
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and return the UCI HAR train/val/test splits.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
        numpy arrays.  Labels are 0-indexed (subtract 1 from the raw 1-indexed
        values so they work cleanly with scikit-learn metrics).
    """
    root = _dataset_root(cfg)
    if not root.exists():
        download_and_extract(cfg)

    X_train_full, y_train_full = _load_split(root, "train")
    X_test, y_test = _load_split(root, "test")

    # Convert labels from 1-indexed to 0-indexed
    y_train_full = y_train_full - 1
    y_test = y_test - 1

    # Fast/smoke-test mode: subsample training data
    if cfg.fast_mode:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(X_train_full), size=min(cfg.fast_n_samples, len(X_train_full)), replace=False)
        X_train_full = X_train_full[idx]
        y_train_full = y_train_full[idx]

    # Deterministic validation split from training data
    n_val = max(1, int(len(X_train_full) * cfg.val_fraction))
    rng = np.random.default_rng(cfg.random_seed)
    val_idx = rng.choice(len(X_train_full), size=n_val, replace=False)
    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_idx] = False

    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    print(
        f"[data] Loaded splits — train: {X_train.shape}, "
        f"val: {X_val.shape}, test: {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
