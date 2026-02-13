# aCare AI/ML - Parkinson's Disease State Detection

ML pipeline and API for detecting Parkinson's patient medication states (ON/OFF) from wearable IMU sensor data.

## Project Structure

```
aCare_AI-ML/
├── backend/                    # FastAPI detection API
│   ├── main.py                 # Application & endpoints
│   ├── detector.py             # ML and rule-based detectors
│   ├── utils.py                # CSV parsing, validation, feature extraction
│   ├── schemas.py              # Pydantic request/response models
│   ├── config.py               # Centralized settings
│   ├── test_api.py             # API tests
│   └── requirements.txt        # Backend-specific dependencies
├── data/
│   ├── EncryptedData/          # Raw encrypted binary sensor data (gitignored)
│   └── DataDescription.md      # Data format documentation
├── Sample_Dataset_Acc/         # Accelerometer-only sample datasets
│   ├── SampleDataset_DataByAI/ # Sample Dataset Genereted using AI
│   ├── SampleDataset_Jan-28/   # First sample (1025 samples, 41s)
│   └── SampleDataset_Jan-29/   # Second sample with training pipeline
├── Sample_Dataset_Acc_Gyro/    # Accelerometer + gyroscope dataset
│   ├── DecodeDataset.py        # Binary-to-CSV decoder
│   ├── DecodeDataset_Train_Test/  # Training pipeline (acc+gyro)
│   └── results/                # Training results
├── requirements.txt            # Project-wide dependencies
└── .gitignore
```

## Quick Start

### 1. Set Up Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2. Run the API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Test Detection

```bash
curl -X POST http://localhost:8080/detect \
  -F "file=@sensor_data.csv"
```

## How It Works

1. **Data Collection** - IMU sensor data (accelerometer + optional gyroscope) is collected from a wrist-worn device via BLE
2. **Decoding** - Binary sensor packets are decoded to CSV using the decoder scripts in `Sample_Dataset_Acc/` or `Sample_Dataset_Acc_Gyro/`
3. **Feature Extraction** - 103 features per window (15 per axis x 6 axes + magnitude/correlation/SVM features) are extracted using sliding windows
4. **Detection** - A trained Random Forest classifier predicts ON/OFF state. Falls back to rule-based heuristics if no model is available

## Detection API

The backend exposes a single detection endpoint:

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| POST   | `/detect` | Upload CSV, get ON/OFF prediction  |
| GET    | `/health` | API health and model status        |
| GET    | `/config` | Current configuration parameters   |

Upload a CSV with accelerometer columns (`acc_x`, `acc_y`, `acc_z`) and optionally gyroscope columns (`gyro_x`, `gyro_y`, `gyro_z`). The API returns the detected state, confidence score, explanation, and sensor statistics.

## Sensor Data Format

CSV files should contain:

| Required Columns | Alternatives                       |
|------------------|------------------------------------|
| `acc_x`          | `accel_x`, `accelerometer_x`, `ax`, `x` |
| `acc_y`          | `accel_y`, `accelerometer_y`, `ay`, `y` |
| `acc_z`          | `accel_z`, `accelerometer_z`, `az`, `z` |

Optional gyroscope columns: `gyro_x`/`gx`, `gyro_y`/`gy`, `gyro_z`/`gz`

**Requirements**: 5-10 seconds of data at ~25-32 Hz sampling rate.

## ML Model

The current model is a Random Forest trained on accelerometer + gyroscope data:
- 206 features (103 raw + 103 baseline z-score deviations)
- Trained with windowed feature extraction (1s windows, 50% overlap)
- Model file: `parkinsons_model_acc_gyro.pkl` (gitignored, placed in `backend/`)

## Sample Datasets

| Dataset | Sensors | Duration | Purpose |
|---------|---------|----------|---------|
| `Sample_Dataset_Acc/Jan-28` | Acc only | 41s | Decoder demonstration |
| `Sample_Dataset_Acc/Jan-29` | Acc only | 10.9s | Training pipeline demo (acc-only) |
| `Sample_Dataset_Acc_Gyro/` | Acc + Gyro | Multiple patients | Full training pipeline (acc+gyro) |

## Dependencies

- Python 3.11+
- FastAPI, uvicorn (API)
- pandas, numpy, scipy (data processing)
- scikit-learn, joblib (ML)
- See [requirements.txt](requirements.txt) for full list

## License

Proprietary - aCare Health Technologies
