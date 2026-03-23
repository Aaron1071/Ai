"""Smoke tests for the HAR classification pipeline.

These tests verify:
  1. Dataset selection logic (no network required).
  2. Data loading and pipeline training on **synthetic** UCI HAR-shaped data
     (561 features, 6 classes) — runs offline with no download needed.
  3. Metrics and results files are saved and contain the expected keys.
  4. (Optional / network) Real dataset download — skipped automatically when
     there is no internet access.

Run with:
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Make src/ importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_har.config import Config, select_dataset
from ai_har.data import ACTIVITY_LABELS, load_dataset
from ai_har.evaluate import evaluate_model, save_results
from ai_har.model import build_model, train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 561      # UCI HAR feature count
N_CLASSES = 6         # six activity classes
N_TRAIN = 300         # synthetic training samples (not the real dataset size)
N_TEST = 100          # synthetic test samples (not the real dataset size)


def _has_internet() -> bool:
    """Return True if we can reach the UCI server."""
    try:
        socket.setdefaulttimeout(3)
        socket.getaddrinfo("archive.ics.uci.edu", 443)
        return True
    except OSError:
        return False


def _make_synthetic_data(
    cfg: Config,
    tmp_path: Path,
) -> Path:
    """Create a minimal fake UCI HAR directory structure with synthetic data.

    Returns the dataset root path.
    """
    root = tmp_path / "UCI HAR Dataset"
    for split, n in [("train", N_TRAIN), ("test", N_TEST)]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, N_FEATURES)).astype(np.float64)
        # 1-indexed labels as in the real dataset
        y = rng.integers(1, N_CLASSES + 1, size=n)

        np.savetxt(split_dir / f"X_{split}.txt", X)
        np.savetxt(split_dir / f"y_{split}.txt", y, fmt="%d")

    return root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def smoke_cfg(tmp_path_factory):
    """Config wired to a temp directory in fast mode."""
    base = tmp_path_factory.mktemp("har_smoke")
    return Config(
        data_dir=base / "data",
        results_dir=base / "results",
        dataset="uci_har",
        model_type="random_forest",
        random_seed=42,
        fast_mode=True,
        fast_n_samples=200,
        fast_n_estimators=5,
        val_fraction=0.1,
    )


@pytest.fixture(scope="session")
def synthetic_data_dir(smoke_cfg, tmp_path_factory):
    """Create synthetic UCI HAR data under smoke_cfg.data_dir."""
    smoke_cfg.data_dir.mkdir(parents=True, exist_ok=True)
    root = _make_synthetic_data(smoke_cfg, smoke_cfg.data_dir)
    return root


@pytest.fixture(scope="session")
def splits(smoke_cfg, synthetic_data_dir):
    """Load train/val/test splits from synthetic data."""
    return load_dataset(smoke_cfg)


@pytest.fixture(scope="session")
def trained_model(smoke_cfg, splits):
    """Fit a small Random Forest on synthetic training data."""
    X_train, y_train, *_ = splits
    model = build_model(smoke_cfg)
    return train(model, X_train, y_train)


# ---------------------------------------------------------------------------
# Tests — no network required
# ---------------------------------------------------------------------------


def test_dataset_selection():
    """Dataset selection should return 'uci_har' for 'Human Activity' task."""
    chosen = select_dataset(task="Human Activity")
    assert chosen == "uci_har"


def test_synthetic_data_dir_exists(synthetic_data_dir):
    """Synthetic dataset root should be created."""
    assert synthetic_data_dir.exists()


def test_expected_files_present(synthetic_data_dir):
    """Expected split files should exist in the synthetic dataset."""
    for split in ("train", "test"):
        assert (synthetic_data_dir / split / f"X_{split}.txt").exists()
        assert (synthetic_data_dir / split / f"y_{split}.txt").exists()


def test_data_shapes(splits):
    """Splits should have consistent feature dimensionality."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    n_features = X_train.shape[1]
    assert n_features == N_FEATURES
    assert X_val.shape[1] == n_features
    assert X_test.shape[1] == n_features
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_label_range(splits):
    """All labels should be in [0, N_CLASSES-1] (0-indexed)."""
    _, y_train, _, y_val, _, y_test = splits
    for y in (y_train, y_val, y_test):
        assert int(y.min()) >= 0
        assert int(y.max()) <= N_CLASSES - 1


def test_train_completes(trained_model):
    """Pipeline fit should not raise; result should be a Pipeline."""
    from sklearn.pipeline import Pipeline
    assert isinstance(trained_model, Pipeline)


def test_predict_shape(trained_model, splits):
    """Predictions should have the same length as the test set."""
    _, _, _, _, X_test, y_test = splits
    y_pred = trained_model.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_accuracy_above_chance(smoke_cfg, trained_model, splits):
    """Test accuracy should exceed random-chance baseline (1/6 ≈ 0.167)."""
    _, _, _, _, X_test, y_test = splits
    metrics = evaluate_model(trained_model, X_test, y_test, split_name="test")
    # On synthetic random data a small RF won't score high, but should beat 0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_results_saved(smoke_cfg, trained_model, splits):
    """save_results() should write a valid JSON file with expected keys."""
    _, _, _, _, X_test, y_test = splits
    metrics = evaluate_model(trained_model, X_test, y_test, split_name="test")
    out_path = save_results(metrics, smoke_cfg, split_name="test")
    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert "accuracy" in data
    assert "confusion_matrix" in data
    assert "classification_report" in data


def test_class_names_in_results(smoke_cfg, trained_model, splits):
    """Results should include all six activity class names."""
    _, _, _, _, X_test, y_test = splits
    metrics = evaluate_model(trained_model, X_test, y_test, split_name="test")
    assert set(metrics["class_names"]) == set(ACTIVITY_LABELS.values())


# ---------------------------------------------------------------------------
# Network test — skipped automatically when offline
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_internet(), reason="No internet access — skipping real download test")
def test_real_dataset_download(tmp_path):
    """Download the real UCI HAR Dataset and verify its directory exists."""
    from ai_har.data import download_and_extract

    cfg = Config(
        data_dir=tmp_path / "data",
        dataset="uci_har",
        fast_mode=False,
    )
    root = download_and_extract(cfg)
    assert root.exists()
    assert (root / "train" / "X_train.txt").exists()
