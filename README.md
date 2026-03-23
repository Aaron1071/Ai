# Human Activity Recognition (HAR) Classification

This project implements a complete **Human Activity Recognition** classification pipeline using the **UCI HAR Dataset** — a widely-used benchmark for sensor-based activity recognition.

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

Each sample contains **561 pre-extracted time and frequency domain features**, making it an ideal reproducible baseline for classification experiments.  The dataset's pre-defined train/test split (70 % / 30 %) ensures fair comparison across studies.

**Dataset selection** is handled automatically: the pipeline selects `uci_har` whenever the task is `"Human Activity"` (see `src/ai_har/config.py → select_dataset()`), matching the screenshot reference _"UCI HAR Dataset: Available"_.

## Project structure

```
Ai/
├── data/                        # Downloaded & extracted dataset (auto-created)
│   └── UCI HAR Dataset/
├── results/                     # Saved metrics & model (auto-created)
│   ├── random_forest_val_results.json
│   ├── random_forest_test_results.json
│   └── random_forest_model.pkl
├── scripts/
│   ├── train.py                 # Main training + evaluation entrypoint
│   └── evaluate.py              # Evaluate a saved model on the test set
├── src/
│   └── ai_har/
│       ├── __init__.py
│       ├── config.py            # Config dataclass + CLI parser + dataset selection
│       ├── data.py              # Download, extract, and load dataset splits
│       ├── model.py             # Model factory (Random Forest / Logistic Regression)
│       └── evaluate.py         # Metrics, confusion matrix, results persistence
├── tests/
│   └── test_smoke.py            # Smoke tests (download + short training run)
├── requirements.txt
└── README.md
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (downloads dataset automatically on first run)

```bash
python scripts/train.py
```

Default settings: **Random Forest (100 trees)**, seed 42, UCI HAR dataset.

### 3. Common CLI options

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

# Custom paths
python scripts/train.py --data-dir /mnt/datasets --results-dir /mnt/outputs
```

### 4. Evaluate a saved model

```bash
python scripts/evaluate.py
```

### 5. Run smoke tests

```bash
pytest tests/test_smoke.py -v
```

> The smoke tests download the real dataset but train on a small subset (300 samples, 5 trees), so they complete quickly.

## Outputs

After training you will find in `results/`:

- `random_forest_val_results.json` — validation accuracy + per-class precision/recall/F1 + confusion matrix
- `random_forest_test_results.json` — same metrics on the held-out test split
- `random_forest_model.pkl` — serialised trained model

## Reproducibility

A fixed `--seed` (default 42) is passed to all random number generators, ensuring identical results across runs on the same platform.

## Expected performance (full dataset, Random Forest 100 trees)

| Metric | Approx. value |
|--------|---------------|
| Test accuracy | ≥ 92 % |
| Macro F1 | ≥ 0.92 |

*(Actual numbers depend on your scikit-learn version and platform.)*
