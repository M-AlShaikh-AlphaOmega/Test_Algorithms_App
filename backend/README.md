# aCare Parkinson's State Detection API

FastAPI backend for detecting Parkinson's patient medication states (ON/OFF) from wearable sensor data.

## Overview

Upload a CSV file with accelerometer (and optionally gyroscope) data. The API returns:
- **Detected State**: `ON`, `OFF`, or `UNKNOWN`
- **Confidence Score**: 0.0 to 1.0
- **Explanation**: Human-readable analysis summary
- **Sensor Statistics**: Data quality metrics

## Project Structure

```
backend/
├── main.py           # FastAPI application & endpoints
├── detector.py       # ML model and rule-based detectors
├── utils.py          # CSV parsing, validation, feature extraction
├── schemas.py        # Pydantic request/response models
├── config.py         # Centralized configuration (Settings)
├── test_api.py       # Pytest tests for the API
├── requirements.txt  # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Or run directly
python main.py
```

### 3. Access the API

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **Health Check**: http://localhost:8080/health

## API Endpoints

| Method | Endpoint  | Description                    |
|--------|-----------|--------------------------------|
| GET    | `/`       | Welcome message & API info     |
| GET    | `/health` | Health check & model status    |
| GET    | `/config` | Current configuration          |
| POST   | `/detect` | Upload CSV for state detection |

## CSV File Format

Your CSV file should contain accelerometer data with these columns:

| Column  | Alternatives                              |
|---------|-------------------------------------------|
| `acc_x` | `accel_x`, `accelerometer_x`, `ax`, `x`  |
| `acc_y` | `accel_y`, `accelerometer_y`, `ay`, `y`  |
| `acc_z` | `accel_z`, `accelerometer_z`, `az`, `z`  |

Optional gyroscope columns:

| Column   | Alternatives                     |
|----------|----------------------------------|
| `gyro_x` | `gyr_x`, `gyroscope_x`, `gx`    |
| `gyro_y` | `gyr_y`, `gyroscope_y`, `gy`    |
| `gyro_z` | `gyr_z`, `gyroscope_z`, `gz`    |

### Requirements
- **Duration**: 5-10 seconds of data
- **Sampling Rate**: ~25Hz (125-320 samples)

### Example CSV

```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
0.12,-0.34,9.75,0.01,-0.02,0.00
0.15,-0.32,9.78,0.01,-0.01,0.01
...
```

## Testing

### Using Swagger UI

1. Navigate to http://localhost:8080/docs
2. Click on `POST /detect`
3. Click "Try it out"
4. Upload your CSV file
5. Click "Execute"

### Using cURL

```bash
curl -X POST "http://localhost:8080/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_sensor_data.csv"
```

### Using Python

```python
import requests

with open("sensor_data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8080/detect",
        files={"file": ("sensor_data.csv", f, "text/csv")}
    )
    print(response.json())
```

### Running Tests

```bash
pytest test_api.py -v
```

## Sample Response

```json
{
  "filename": "patient_001_session.csv",
  "detected_state": "ON",
  "confidence": 0.82,
  "explanation": "ML model detected ON state with 82% confidence based on 5 analysis window(s).",
  "sensor_stats": {
    "sample_count": 250,
    "duration_seconds": 10.0,
    "mean_magnitude": 9.85,
    "std_magnitude": 0.45,
    "movement_intensity": 0.35
  }
}
```

## Detection Pipeline

The `/detect` endpoint runs this pipeline:

1. **Parse CSV** - Decode file content, validate structure
2. **Validate Data** - Find sensor columns, check data quality, handle missing values
3. **Compute Statistics** - Calculate magnitude, activity levels, zero-crossing rates
4. **Run Detection** - ML model (if loaded) or rule-based fallback

### ML Detector (`MLDetector`)
- Loads a trained Random Forest from `parkinsons_model_acc_gyro.pkl`
- Extracts 103 features per window (15 per axis x 6 axes + cross-axis features)
- Appends baseline z-score deviations (103 more features, 206 total)
- Windows: 1-second duration, 50% overlap, averaged across all windows

### Rule-Based Fallback (`RuleBasedDetector`)
- Used when no `.pkl` model file is available
- Applies simplified heuristics based on movement intensity, tremor indicators (zero-crossing rate), and movement variability
- Not intended for clinical use

## Configuration

Environment variables (prefix with `ACARE_`):

| Variable                          | Default | Description                    |
|-----------------------------------|---------|--------------------------------|
| `ACARE_DEBUG`                     | `true`  | Enable debug mode              |
| `ACARE_EXPECTED_SAMPLE_RATE_HZ`  | `25`    | Expected sensor sampling rate  |
| `ACARE_MIN_SAMPLES`              | `125`   | Minimum samples (~5 sec)       |
| `ACARE_MAX_SAMPLES`              | `320`   | Maximum samples (~10 sec)      |
| `ACARE_CONFIDENCE_THRESHOLD`     | `0.6`   | Below this returns UNKNOWN     |
| `ACARE_MODEL_PATH`               | `parkinsons_model_acc_gyro.pkl` | Model file path |

## Error Handling

| HTTP Code | Error Type             | Description                        |
|-----------|------------------------|------------------------------------|
| 400       | `invalid_file_type`    | File is not a CSV                  |
| 400       | `empty_file`           | Uploaded file is empty             |
| 422       | `data_validation_error`| Missing columns, insufficient data |
| 500       | `processing_error`     | Unexpected server error            |

## License

Proprietary - aCare Health Technologies
