# Human Activity Recognition (HAR) Classification

This project implements a complete **Human Activity Recognition** classification pipeline using the **UCI HAR Dataset** — a widely-used benchmark for sensor-based activity recognition.

Two training paths are provided:

| Path | Script | Model | Input |
|------|--------|-------|-------|
| **Classic ML** | `scripts/train.py` | Random Forest / Logistic Regression | 561 pre-extracted features per window |
| **Deep Learning** | `scripts/train_dl.py` | 1D CNN (PyTorch) | Raw 9-channel × 128-timestep inertial signals |

## Why UCI HAR?

The UCI HAR Dataset records smartphone accelerometer and gyroscope signals from 30 volunteers performing six daily activities:

| Code | Activity |
|------|----------|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

Each sample contains **561 pre-extracted time and frequency domain features** (used by the classic ML path) as well as the **raw inertial signals** (9 channels × 128 timesteps, used by the deep-learning path), making it an ideal reproducible benchmark.  The dataset's pre-defined train/test split (70 % / 30 %) ensures fair comparison across studies.

## Project structure

```
Ai/
├── data/                        # Downloaded & extracted dataset (auto-created)
│   └── UCI HAR Dataset/
├── results/                     # Saved metrics & model (auto-created)
│   ├── random_forest_val_results.json
│   ├── random_forest_test_results.json
│   ├── random_forest_model.pkl
│   ├── cnn1d_val_results.json
│   ├── cnn1d_test_results.json
│   └── cnn1d_model.pt
├── scripts/
│   ├── train.py                 # Classic ML training entrypoint
│   ├── train_dl.py              # Deep-learning (1D CNN) training entrypoint
│   └── evaluate.py              # Evaluate a saved ML model on the test set
├── src/
│   └── ai_har/
│       ├── __init__.py
│       ├── config.py            # Config dataclass + CLI parser + dataset selection
│       ├── data.py              # Download, extract, load features & raw signals
│       ├── model.py             # Scikit-learn model factory (RF / LR)
│       ├── dl_model.py          # PyTorch CNN1D, training loop, save/load
│       └── evaluate.py          # Metrics, confusion matrix, results persistence
├── tests/
│   └── test_smoke.py            # Smoke tests (ML + DL paths, no download needed)
├── requirements.txt
└── README.md
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2a. Deep-learning path — 1D CNN on raw inertial signals

```bash
# Quick demo (≈ 500 training samples, 3 epochs — runs in seconds on CPU):
python scripts/train_dl.py --fast

# Standard run (full dataset, 10 epochs):
python scripts/train_dl.py

# Custom hyper-parameters:
python scripts/train_dl.py --epochs 5 --batch-size 64 --lr 1e-3
```

**DL CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--seed` | `42` | Random seed |
| `--data-dir` | `data` | Dataset cache directory |
| `--results-dir` | `results` | Output directory |
| `--fast` | off | Demo mode (500 samples, 3 epochs) |

### 2b. Classic ML path — Random Forest / Logistic Regression

```bash
python scripts/train.py
```

Default settings: **Random Forest (100 trees)**, seed 42, UCI HAR dataset.

**ML CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `random_forest` | `random_forest` or `logistic_regression` |
| `--n-estimators` | `100` | Number of trees (Random Forest) |
| `--seed` | `42` | Random seed |
| `--data-dir` | `data` | Where to cache the dataset |
| `--results-dir` | `results` | Where to save outputs |
| `--fast` | off | Smoke-test mode (500 samples, 10 trees) |

```bash
# Logistic Regression
python scripts/train.py --model logistic_regression

# Fast smoke-test run
python scripts/train.py --fast
```

### 3. Evaluate a saved ML model

```bash
python scripts/evaluate.py
```

### 4. Run smoke tests

```bash
pytest tests/test_smoke.py -v
```

> The smoke tests cover both the ML and DL paths using synthetic data — no download or GPU required.

## Outputs

After training you will find in `results/`:

**Classic ML**
- `random_forest_val_results.json` — validation accuracy + per-class precision/recall/F1 + confusion matrix
- `random_forest_test_results.json` — same metrics on the held-out test split
- `random_forest_model.pkl` — serialised trained model

**Deep Learning**
- `cnn1d_val_results.json` — validation metrics
- `cnn1d_test_results.json` — test metrics
- `cnn1d_model.pt` — model weights (PyTorch `state_dict`)

## Reproducibility

A fixed `--seed` (default 42) is passed to all random number generators, ensuring identical results across runs on the same platform.

## Expected performance (full dataset)

| Path | Model | Test accuracy |
|------|-------|--------------|
| Classic ML | Random Forest 100 trees | ≥ 92 % |
| Deep Learning | 1D CNN, 10 epochs | ≥ 88 % |

*(Actual numbers depend on your library versions, platform, and random seed.)*

