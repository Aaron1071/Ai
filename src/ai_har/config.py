"""Configuration and dataset selection for the HAR project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Supported datasets – extend this dict to add more sources
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, Dict] = {
    "uci_har": {
        "name": "UCI HAR Dataset",
        "task": "Human Activity",
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00240/UCI%20HAR%20Dataset.zip"
        ),
        "zip_name": "UCI_HAR_Dataset.zip",
        "extract_subdir": "UCI HAR Dataset",
        "description": (
            "UCI HAR Dataset: sensor data from 30 volunteers performing six daily "
            "activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, "
            "STANDING, LAYING) captured via smartphone accelerometer and gyroscope."
        ),
        # Set to False to exclude a dataset from automatic selection without
        # removing its entry from the registry (e.g. during maintenance).
        "available": True,
    }
}

# Default dataset chosen for the Human Activity task
DEFAULT_DATASET = "uci_har"


@dataclass
class Config:
    """Central configuration object."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    # Dataset selection
    dataset: str = DEFAULT_DATASET

    # Model
    model_type: str = "random_forest"  # "random_forest" | "logistic_regression"
    n_estimators: int = 100           # trees for Random Forest
    max_iter: int = 1000              # iterations for Logistic Regression
    random_seed: int = 42

    # Validation split (fraction of training data held out for validation)
    val_fraction: float = 0.1

    # Smoke-test / fast mode
    fast_mode: bool = False
    fast_n_samples: int = 500
    fast_n_estimators: int = 10

    # -----------------------------------------------------------------------
    def dataset_info(self) -> Dict:
        if self.dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Choose from: {list(DATASET_REGISTRY.keys())}"
            )
        return DATASET_REGISTRY[self.dataset]

    def effective_n_estimators(self) -> int:
        return self.fast_n_estimators if self.fast_mode else self.n_estimators


def select_dataset(task: str = "Human Activity") -> str:
    """Return the dataset key best suited for *task*.

    This implements the dataset-selection logic referenced in the project spec.
    Currently only one dataset is registered; extending DATASET_REGISTRY will
    allow this function to pick among multiple options automatically.
    """
    candidates = [
        key
        for key, meta in DATASET_REGISTRY.items()
        if meta["task"].lower() == task.lower() and meta["available"]
    ]
    if not candidates:
        raise ValueError(f"No available dataset found for task: '{task}'")
    chosen = candidates[0]
    info = DATASET_REGISTRY[chosen]
    print(f"[dataset-selection] Task='{task}' → {info['name']}: Available")
    return chosen


def build_config_from_args(argv=None) -> Config:
    """Parse CLI arguments and return a populated :class:`Config`."""
    parser = argparse.ArgumentParser(
        description="HAR classification pipeline (UCI HAR Dataset)"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to use (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        dest="model_type",
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Classifier type (default: %(default)s)",
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
        "--seed",
        type=int,
        default=42,
        dest="random_seed",
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees (Random Forest, default: %(default)s)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        dest="fast_mode",
        help="Fast/smoke-test mode: small subset + fewer trees",
    )
    args = parser.parse_args(argv)
    return Config(
        dataset=args.dataset,
        model_type=args.model_type,
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        random_seed=args.random_seed,
        n_estimators=args.n_estimators,
        fast_mode=args.fast_mode,
    )
