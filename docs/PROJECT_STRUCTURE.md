# Project Structure

## Overview

`acare-ml` is a production-grade ML pipeline for Parkinson's disease detection from IMU sensor data. This document explains the repository layout, typical workflows, and development conventions.

## Repository Layout

| Path | Purpose | Examples |
|------|---------|----------|
| `configs/` | YAML configuration files for each pipeline stage | `dataset.yaml`, `training.yaml` |
| `data/raw/` | Original, immutable data files | `subject_001.csv`, `subject_002.csv` |
| `data/interim/` | Intermediate data after basic cleaning | `cleaned_imu_data.csv` |
| `data/processed/` | Final feature matrices ready for training | `train_features.csv`, `test_features.csv` |
| `artifacts/models/` | Trained model files (pkl, joblib, onnx) | `model_v1.pkl`, `best_model.joblib` |
| `artifacts/reports/` | Performance metrics, classification reports | `eval_metrics.json`, `confusion_matrix.png` |
| `artifacts/figures/` | Plots and visualizations | `feature_importance.png` |
| `notebooks/` | Jupyter notebooks for exploration | `01_eda.ipynb`, `02_feature_analysis.ipynb` |
| `scripts/` | One-off scripts and utilities | `download_data.py`, `export_model.py` |
| `src/acare_ml/` | Main package source code | See module breakdown below |
| `tests/` | Unit and integration tests | `test_features.py`, `test_models.py` |

### Module Breakdown (`src/acare_ml/`)

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `common/` | Shared utilities (logging, config loading) | `logging.py`, `config.py` |
| `domain/` | Domain constants and business logic | `constants.py`, `activities.py`, `clinical_thresholds.py` |
| `dataio/` | Data readers for various file formats | `readers.py` (CSV, Parquet, custom formats) |
| `preprocessing/` | Signal processing and cleaning transforms | `transforms.py` (normalization, filtering) |
| `features/` | Feature extraction logic | `extractors.py` (statistical, frequency, time-domain) |
| `models/` | Model definitions and base classes | `base.py`, `random_forest.py` |
| `training/` | Training loops, validation, hyperparameter tuning | `trainer.py` |
| `evaluation/` | Model evaluation and metrics | `metrics.py`, `cross_validator.py`, `reports.py` |
| `validation/` | Data validation and quality checks | `schema.py`, `quality.py`, `clinical.py` |
| `subjects/` | Subject-level operations and splitting | `splitter.py`, `metadata.py`, `aggregation.py` |
| `pipelines/` | End-to-end orchestration | `dataset_pipeline.py`, `feature_pipeline.py`, `training_pipeline.py`, `inference.py` |
| `serving/` | Model serving and API endpoints | `api.py`, `predictor.py` |

## Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  Raw Data   │────▶│ build-dataset│────▶│build-features │────▶│  train   │
│ data/raw/   │     │ data/interim/│     │data/processed/│     │artifacts/│
└─────────────┘     └──────────────┘     └───────────────┘     └──────────┘
                                                                      │
                                                                      ▼
                                                                 ┌─────────┐
                                                                 │  infer  │
                                                                 └─────────┘
```

1. **build-dataset**: Load raw CSVs, validate schema, handle missing data → `data/interim/`
2. **build-features**: Window segmentation, feature extraction → `data/processed/`
3. **train**: Split data, train model, save artifacts → `artifacts/models/`
4. **infer**: Load model, predict on new data, save predictions

## Typical Workflow

### 1. Setup

```bash
# Clone repo and install
git clone <repo-url>
cd acare-ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Add Raw Data

Place raw IMU CSV files in `data/raw/`:
```
data/raw/
  ├── subject_001.csv
  ├── subject_002.csv
  └── labels.csv
```

### 3. Run Pipeline

```bash
# Step 1: Build dataset (clean, merge labels)
acare-ml build-dataset --config configs/dataset.yaml --output-dir data/interim

# Step 2: Extract features (windowing + feature engineering)
acare-ml build-features --config configs/features.yaml \
  --input-dir data/interim --output-dir data/processed

# Step 3: Train model
acare-ml train --config configs/training.yaml \
  --data-dir data/processed --output-dir artifacts/models

# Step 4: Inference on new data
acare-ml infer --model-path artifacts/models/model.pkl \
  --data-path data/processed/test.csv --output-path artifacts/predictions.csv
```

### 4. Test and Validate

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format
```

## Where to Add New Code

### Adding a New Dataset Reader

1. Create function in `src/acare_ml/dataio/readers.py`:
   ```python
   def read_parquet_imu(file_path: Path) -> pd.DataFrame:
       return pd.read_parquet(file_path)
   ```
2. Update `configs/dataset.yaml` with new file pattern

### Adding New Features

1. Add extractor in `src/acare_ml/features/extractors.py`:
   ```python
   def extract_frequency_features(window: pd.DataFrame) -> dict:
       # FFT, spectral features, etc.
       return features
   ```
2. Register in `configs/features.yaml`:
   ```yaml
   feature_extractors:
     - statistical
     - frequency  # new
   ```

### Adding a New Model

1. Create model class in `src/acare_ml/models/`:
   ```python
   # src/acare_ml/models/xgboost_model.py
   from .base import BaseModel
   class XGBoostModel(BaseModel):
       def fit(self, X, y): ...
       def predict(self, X): ...
   ```
2. Update `configs/training.yaml`:
   ```yaml
   model_type: xgboost
   hyperparameters:
     n_estimators: 200
   ```

## Configuration Guide

### `configs/dataset.yaml`
- Controls raw data ingestion
- Specifies subject ID column, label column
- File patterns and data directory paths

### `configs/features.yaml`
- Window size, overlap ratio, sampling rate
- Which feature extractors to apply
- Input/output directories

### `configs/training.yaml`
- Model type and hyperparameters
- Train/test split ratio
- Random seed for reproducibility
- Output paths for models and reports

### `configs/inference.yaml`
- Model path to load
- Input data path
- Output predictions path
- Batch size for inference

## Testing & Quality

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_features.py

# With coverage
pytest --cov=acare_ml --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## Common Pitfalls

### 1. Data Leakage

**Problem**: Using future information in features (e.g., normalizing with statistics from entire dataset).

**Solution**:
- Compute normalization parameters on training set only
- Apply same parameters to validation/test sets
- Use subject-level splits (not random row splits)

### 2. Inconsistent Sampling Rates

**Problem**: Mixing data from different devices with different sampling rates.

**Solution**:
- Standardize all data to `SAMPLING_RATE` in `domain/constants.py`
- Resample or interpolate during `build-dataset` stage
- Document sampling rate in metadata

### 3. Mixing Raw and Processed Data

**Problem**: Accidentally using raw data for training instead of processed features.

**Solution**:
- Strictly follow `raw → interim → processed → artifacts` flow
- Never point training scripts directly to `data/raw/`
- Use config files to enforce correct paths

### 4. Subject-Level Splits

**Problem**: Random splits can leak information when subjects have multiple recordings.

**Solution**:
- Split by subject ID, not by rows
- Example: subjects 1-80 train, 81-100 test
- Prevents seeing same subject in both train and test

### 5. Not Saving Preprocessing Steps

**Problem**: Training model with features but forgetting how to replicate preprocessing for inference.

**Solution**:
- Save preprocessing pipeline alongside model (e.g., `sklearn.Pipeline`)
- Or save config used to generate features with model metadata
- Document feature extraction steps in `artifacts/reports/`

## Next Steps

- See `README.md` for quickstart commands
- Check `notebooks/` for exploratory examples
- Review `tests/` for usage patterns
- Read inline code comments for implementation details
