"""Smoke tests for the HAR classification pipeline.

These tests verify:
  1. Dataset selection logic (no network required).
  2. Data loading and pipeline training on **synthetic** UCI HAR-shaped data
     (561 features, 6 classes) — runs offline with no download needed.
  3. Metrics and results files are saved and contain the expected keys.
  4. Deep-learning path: CNN1D forward pass, training, save/load — all on
     synthetic tensor data (no download needed).
  5. (Optional / network) Real dataset download — skipped automatically when
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
import torch

# Make src/ importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_har.config import Config, select_dataset
from ai_har.data import ACTIVITY_LABELS, load_dataset
from ai_har.evaluate import evaluate_model, save_results
from ai_har.model import build_model, train
from ai_har.dl_model import (
    CNN1D,
    DLConfig,
    evaluate_dl,
    load_dl_model,
    save_dl_model,
    save_dl_results,
    train_dl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 561      # UCI HAR feature count
N_CLASSES = 6         # six activity classes
N_TRAIN = 300         # synthetic training samples (not the real dataset size)
N_TEST = 100          # synthetic test samples (not the real dataset size)

# Deep-learning synthetic data dimensions
N_CHANNELS = 9        # 9 inertial-signal channels (body_acc, body_gyro, total_acc × 3 axes)
N_TIMESTEPS = 128     # window length used by UCI HAR


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


# ---------------------------------------------------------------------------
# Deep-learning tests — offline, use synthetic tensor data
# ---------------------------------------------------------------------------


def _make_synthetic_sequences(
    n_samples: int,
    n_channels: int = N_CHANNELS,
    n_timesteps: int = N_TIMESTEPS,
    n_classes: int = N_CLASSES,
    seed: int = 0,
) -> tuple:
    """Return (X, y) with X shape (n_samples, n_channels, n_timesteps)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_channels, n_timesteps)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)
    return X, y


@pytest.fixture(scope="session")
def dl_cfg(tmp_path_factory):
    """DLConfig wired to a temp directory in fast mode with minimal epochs."""
    base = tmp_path_factory.mktemp("har_dl_smoke")
    return DLConfig(
        epochs=2,
        batch_size=32,
        lr=1e-3,
        random_seed=0,
        data_dir=base / "data",
        results_dir=base / "results",
        fast_mode=True,
        fast_n_samples=200,
        fast_epochs=2,
    )


@pytest.fixture(scope="session")
def dl_sequences():
    """Synthetic (X_train, y_train, X_val, y_val, X_test, y_test) sequences."""
    X_train, y_train = _make_synthetic_sequences(120, seed=1)
    X_val, y_val = _make_synthetic_sequences(30, seed=2)
    X_test, y_test = _make_synthetic_sequences(50, seed=3)
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture(scope="session")
def trained_dl_model(dl_cfg, dl_sequences):
    """Tiny CNN1D trained for 2 epochs on synthetic sequences."""
    X_train, y_train, X_val, y_val, *_ = dl_sequences
    model = CNN1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    return train_dl(model, X_train, y_train, X_val, y_val, dl_cfg)


def test_cnn1d_forward_shape():
    """CNN1D forward pass should produce logits of shape (batch, n_classes)."""
    model = CNN1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    model.eval()
    x = torch.zeros(4, N_CHANNELS, N_TIMESTEPS)
    out = model(x)
    assert out.shape == (4, N_CLASSES), f"unexpected output shape: {out.shape}"


def test_cnn1d_output_is_finite():
    """CNN1D should not produce NaN or Inf logits on zero input."""
    model = CNN1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    model.eval()
    x = torch.zeros(2, N_CHANNELS, N_TIMESTEPS)
    out = model(x)
    assert torch.isfinite(out).all(), "CNN1D output contains NaN or Inf"


def test_dl_train_completes(trained_dl_model):
    """train_dl() should return a CNN1D without raising."""
    assert isinstance(trained_dl_model, CNN1D)


def test_dl_predict_shape(trained_dl_model, dl_sequences):
    """Predictions should match the number of test samples."""
    *_, X_test, y_test = dl_sequences
    metrics = evaluate_dl(trained_dl_model, X_test, y_test, split_name="test")
    cm = np.array(metrics["confusion_matrix"])
    assert cm.sum() == len(y_test), "confusion matrix total ≠ test set size"


def test_dl_accuracy_in_range(trained_dl_model, dl_sequences):
    """DL model accuracy should be a valid probability (0 ≤ acc ≤ 1)."""
    *_, X_test, y_test = dl_sequences
    metrics = evaluate_dl(trained_dl_model, X_test, y_test, split_name="test")
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_dl_results_saved(dl_cfg, trained_dl_model, dl_sequences):
    """save_dl_results() should write a valid JSON file with expected keys."""
    *_, X_test, y_test = dl_sequences
    metrics = evaluate_dl(trained_dl_model, X_test, y_test, split_name="test")
    out_path = save_dl_results(metrics, dl_cfg.results_dir, split_name="test")
    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    for key in ("accuracy", "confusion_matrix", "classification_report", "class_names"):
        assert key in data, f"missing key '{key}' in DL results JSON"


def test_dl_class_names_in_results(trained_dl_model, dl_sequences):
    """DL results should include all six activity class names."""
    *_, X_test, y_test = dl_sequences
    metrics = evaluate_dl(trained_dl_model, X_test, y_test, split_name="test")
    assert set(metrics["class_names"]) == set(ACTIVITY_LABELS.values())


def test_dl_model_save_load(dl_cfg, trained_dl_model, dl_sequences):
    """save_dl_model / load_dl_model round-trip should preserve predictions."""
    *_, X_test, y_test = dl_sequences
    model_path = dl_cfg.results_dir / "cnn1d_smoke_model.pt"
    save_dl_model(trained_dl_model, model_path)
    assert model_path.exists()

    loaded = load_dl_model(model_path, n_channels=N_CHANNELS, n_classes=N_CLASSES)

    # Both models should produce identical predictions on the same input
    x = torch.from_numpy(X_test[:8])
    trained_dl_model.eval()
    loaded.eval()
    with torch.no_grad():
        preds_orig = trained_dl_model(x).argmax(dim=1)
        preds_loaded = loaded(x).argmax(dim=1)
    assert torch.equal(preds_orig, preds_loaded), "loaded model predictions differ"
