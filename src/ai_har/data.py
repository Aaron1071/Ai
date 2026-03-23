"""Dataset download, extraction, and loading for the UCI HAR Dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Tuple
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

# Ordered list of the 9 inertial-signal channel names (3 sensors × 3 axes)
INERTIAL_SIGNAL_FILES: List[str] = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


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


def load_inertial_signals(
    cfg: Config,
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw inertial-signal windows for *split* ('train' or 'test').

    Each UCI HAR window contains 128 consecutive timesteps sampled at 50 Hz
    from 9 sensor channels (body_acc, body_gyro, total_acc × 3 axes).

    Parameters
    ----------
    cfg:
        Project configuration object.
    split:
        Either ``'train'`` or ``'test'``.

    Returns
    -------
    X : np.ndarray, shape ``(N, 9, 128)`` — float32
        9-channel × 128-timestep windows arranged for PyTorch ``Conv1d``
        (channels-first convention).
    y : np.ndarray, shape ``(N,)`` — int64
        0-indexed activity labels (raw 1-indexed values minus 1).
    """
    root = _dataset_root(cfg)
    if not root.exists():
        download_and_extract(cfg)

    signals_dir = root / split / "Inertial Signals"
    channels = []
    for sig_name in INERTIAL_SIGNAL_FILES:
        path = signals_dir / f"{sig_name}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Inertial signal file not found: '{path}'. "
                "Ensure the UCI HAR Dataset was fully downloaded and extracted "
                "with download_and_extract() — the Inertial Signals folder must be present."
            )
        arr = np.loadtxt(path)  # shape (N, 128)
        channels.append(arr)

    # Stack channels: (N, 128) × 9 → (N, 9, 128) via intermediate (N, 128, 9)
    X = np.stack(channels, axis=-1).transpose(0, 2, 1)  # (N, 9, 128)

    y_path = root / split / f"y_{split}.txt"
    y = np.loadtxt(y_path, dtype=int) - 1  # convert to 0-indexed

    return X.astype(np.float32), y.astype(np.int64)
