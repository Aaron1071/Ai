"""Deep-learning components for HAR: 1D CNN, training loop, save/load.

The model targets the raw UCI HAR inertial-signal sequences
(9 channels × 128 timesteps) and is intentionally lightweight so that it
trains in a few minutes on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai_har.data import ACTIVITY_LABELS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DLConfig:
    """Hyper-parameters for the deep-learning training path."""

    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    random_seed: int = 42
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    # Fast / demo mode
    fast_mode: bool = False
    fast_n_samples: int = 500
    fast_epochs: int = 3

    def effective_epochs(self) -> int:
        return self.fast_epochs if self.fast_mode else self.epochs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CNN1D(nn.Module):
    """Lightweight 1D CNN for HAR on 9-channel × 128-timestep windows.

    Architecture
    ------------
    Two ``Conv1d`` blocks (16 → 32 filters) each followed by ReLU and
    ``MaxPool1d(2)``, reducing the sequence length from 128 to 32.  Global
    average pooling then collapses the time dimension, and a single linear
    layer maps the 32-dimensional representation to ``n_classes`` logits.

    Parameters
    ----------
    n_channels:
        Number of input sensor channels (default: 9 for UCI HAR).
    n_classes:
        Number of activity classes (default: 6).
    """

    def __init__(self, n_channels: int = 9, n_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                              # 128 → 64
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                              # 64 → 32
        )
        self.pool = nn.AdaptiveAvgPool1d(1)               # 32 → 1
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, n_channels, timesteps)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_classes)``.
        """
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _compute_accuracy(
    model: CNN1D,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(dim=1)
            correct += (preds == y_batch.to(device)).sum().item()
            total += len(y_batch)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_dl(
    model: CNN1D,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dl_cfg: DLConfig,
    device: Optional[torch.device] = None,
) -> CNN1D:
    """Train *model* and return the weights with the best validation accuracy.

    Parameters
    ----------
    model:
        Untrained :class:`CNN1D` instance.
    X_train, y_train:
        Training data; ``X_train`` shape ``(N, 9, 128)``.
    X_val, y_val:
        Validation data; same channel/timestep convention.
    dl_cfg:
        Training hyper-parameters.
    device:
        Target device.  Defaults to CUDA if available, otherwise CPU.

    Returns
    -------
    CNN1D
        Model loaded with the best-validation-accuracy weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dl_cfg.lr)
    criterion = nn.CrossEntropyLoss()
    epochs = dl_cfg.effective_epochs()

    train_loader = _make_dataloader(X_train, y_train, dl_cfg.batch_size)
    val_loader = _make_dataloader(X_val, y_val, dl_cfg.batch_size, shuffle=False)

    best_val_acc: float = -1.0
    best_state: Optional[Dict] = None

    print(f"[dl] Training CNN1D on {device} for {epochs} epoch(s) …")
    for epoch in range(1, epochs + 1):
        # --- train phase ---
        model.train()
        running_loss = 0.0
        total_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)

        # --- validation phase ---
        val_acc = _compute_accuracy(model, val_loader, device)
        avg_loss = running_loss / max(total_samples, 1)
        print(
            f"  epoch {epoch:>{len(str(epochs))}d}/{epochs}"
            f" | train loss: {avg_loss:.4f}"
            f" | val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[dl] Best validation accuracy: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_dl(
    model: CNN1D,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str = "test",
    device: Optional[torch.device] = None,
) -> Dict:
    """Compute classification metrics for *model* on ``(X, y_true)``.

    Returns
    -------
    dict
        Contains ``accuracy``, ``classification_report``, ``confusion_matrix``,
        and ``class_names`` — the same schema used by the scikit-learn path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    loader = _make_dataloader(X, y_true, batch_size=256, shuffle=False)

    all_preds: list = []
    all_true: list = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(y_batch.numpy().tolist())

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = np.array(all_preds)
    y_true_arr = np.array(all_true)
    class_names = list(ACTIVITY_LABELS.values())

    acc = accuracy_score(y_true_arr, y_pred)
    report = classification_report(
        y_true_arr,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred).tolist()

    print(f"\n[dl-evaluate] {split_name.upper()} accuracy: {acc:.4f}")
    print(
        classification_report(
            y_true_arr,
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


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_dl_model(model: CNN1D, path: Path) -> None:
    """Persist *model* weights to *path* using :func:`torch.save`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[dl] Saved model weights to '{path}'")


def load_dl_model(
    path: Path,
    n_channels: int = 9,
    n_classes: int = 6,
) -> CNN1D:
    """Load :class:`CNN1D` weights from *path*.

    Parameters
    ----------
    path:
        File written by :func:`save_dl_model`.
    n_channels:
        Must match the value used when the model was originally created.
    n_classes:
        Must match the value used when the model was originally created.
    """
    model = CNN1D(n_channels=n_channels, n_classes=n_classes)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"[dl] Loaded model weights from '{path}'")
    return model


def save_dl_results(
    metrics: Dict,
    results_dir: Path,
    split_name: str = "test",
) -> Path:
    """Persist *metrics* dict as a JSON file inside *results_dir*."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"cnn1d_{split_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dl-evaluate] Results saved to '{out_path}'")
    return out_path
