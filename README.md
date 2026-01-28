# acare-ml

Production-ready ML pipeline for Parkinson's disease detection from IMU sensor data.

## Directory Structure

```
acare-ml/
├── configs/              # Pipeline configuration files
│   ├── dataset.yaml      # Data ingestion settings
│   ├── features.yaml     # Feature extraction config
│   ├── training.yaml     # Model training hyperparameters
│   └── inference.yaml    # Inference settings
├── data/
│   ├── raw/              # Original immutable data files
│   ├── interim/          # Intermediate cleaned data
│   └── processed/        # Final feature matrices
├── artifacts/
│   ├── models/           # Trained model files (.pkl, .joblib)
│   ├── reports/          # Performance metrics and eval reports
│   └── figures/          # Plots and visualizations
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # One-off utility scripts
├── src/acare_ml/         # Main package source
│   ├── common/           # Logging, config utilities
│   ├── domain/           # Business logic, constants, clinical thresholds
│   ├── dataio/           # Data readers
│   ├── preprocessing/    # Signal processing transforms
│   ├── features/         # Feature extractors
│   ├── models/           # Model definitions
│   ├── training/         # Training logic
│   ├── evaluation/       # Metrics, cross-validation, reports
│   ├── validation/       # Data validation, quality checks
│   ├── subjects/         # Subject-level operations and splitting
│   ├── pipelines/        # End-to-end workflows
│   └── serving/          # Model serving/API
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
│   └── PROJECT_STRUCTURE.md  # Detailed structure guide
├── pyproject.toml        # Package metadata and dependencies
├── Makefile              # Development commands
├── pytest.ini            # Pytest configuration
└── .env.example          # Environment variables template
```

## Quick Start

### Installation

```bash
# Clone and navigate
git clone <repo-url>
cd acare-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check CLI is accessible
acare-ml --help

# Or via module
python -m acare_ml.cli --help
```

### Run Pipeline

```bash
# 1. Build dataset from raw data
acare-ml build-dataset \
  --config configs/dataset.yaml \
  --output-dir data/interim

# 2. Extract features
acare-ml build-features \
  --config configs/features.yaml \
  --input-dir data/interim \
  --output-dir data/processed

# 3. Train model
acare-ml train \
  --config configs/training.yaml \
  --data-dir data/processed \
  --output-dir artifacts/models

# 4. Run inference
acare-ml infer \
  --model-path artifacts/models/model.pkl \
  --data-path data/processed/test_features.csv \
  --output-path artifacts/predictions.csv
```

## Configuration Files

| Config | Purpose |
|--------|---------|
| `dataset.yaml` | Raw data paths, file patterns, subject/label columns |
| `features.yaml` | Window size, overlap, sampling rate, feature extractors |
| `training.yaml` | Model type, hyperparameters, train/test split, random seed |
| `inference.yaml` | Model path, input data, output path, batch size |

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed configuration guides.

## Artifacts

All outputs are saved in the `artifacts/` directory:

- **models/**: Serialized model files (`.pkl`, `.joblib`, `.onnx`)
- **reports/**: JSON metrics, classification reports, evaluation results
- **figures/**: Feature importance plots, confusion matrices, ROC curves

These artifacts are gitignored by default. Use a model registry or DVC for version control.

## Development

### Testing

```bash
# Run all tests
make test

# Or directly with pytest
pytest

# With coverage report
pytest --cov=acare_ml --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Lint
make lint

# Or use tools directly
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Makefile Targets

```bash
make help      # Show available targets
make install   # Install package with dev dependencies
make test      # Run tests
make lint      # Run linter
make format    # Format code
make clean     # Remove build artifacts
```

## Documentation

- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**: Detailed guide on folder purposes, module breakdown, workflow, and best practices
- **Inline code comments**: Implementation details in source files
- **Notebooks**: Exploratory analysis examples in `notebooks/`

## Key Concepts

### Data Flow

```
raw → interim → processed → artifacts
```

1. **raw**: Original sensor data (immutable)
2. **interim**: Cleaned, validated data
3. **processed**: Feature matrices ready for ML
4. **artifacts**: Models, metrics, predictions

### Avoiding Data Leakage

- Use **subject-level splits** (not random row splits)
- Compute normalization/scaling on **training data only**
- Never use test data for feature engineering decisions
- See [Common Pitfalls](docs/PROJECT_STRUCTURE.md#common-pitfalls) for details

## Requirements

- Python 3.11+
- Dependencies: numpy, pandas, scikit-learn, click, pyyaml
- Dev tools: pytest, black, ruff, mypy

## License

MIT

## Contributing

1. Follow existing code structure
2. Add tests for new features
3. Run `make format` and `make lint` before committing
4. Update documentation as needed
