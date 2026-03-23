"""Model factory and training for the HAR classification pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_har.config import Config

Classifier = Union[RandomForestClassifier, LogisticRegression]


def build_model(cfg: Config) -> Pipeline:
    """Construct a scikit-learn Pipeline for the chosen classifier type."""
    if cfg.model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=cfg.effective_n_estimators(),
            random_state=cfg.random_seed,
            n_jobs=-1,
        )
        # UCI HAR features are already normalised; scaler is a no-op here but
        # keeps the pipeline consistent for any raw-signal variant.
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if cfg.model_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=cfg.max_iter,
            random_state=cfg.random_seed,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    raise ValueError(
        f"Unknown model_type '{cfg.model_type}'. "
        "Choose 'random_forest' or 'logistic_regression'."
    )


def train(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Fit *model* on the training data and return it."""
    print(f"[model] Training {model.named_steps['clf'].__class__.__name__} …")
    model.fit(X_train, y_train)
    print("[model] Training complete.")
    return model


def save_model(model: Pipeline, path: Path) -> None:
    """Persist *model* to *path* using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[model] Saved model to '{path}'")


def load_model(path: Path) -> Pipeline:
    """Load a previously saved model from *path*."""
    with open(path, "rb") as f:
        model = pickle.load(f)  # noqa: S301
    print(f"[model] Loaded model from '{path}'")
    return model
