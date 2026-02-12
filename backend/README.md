# aCare Parkinson's State Detection API

A FastAPI backend for detecting Parkinson's patient medication states (ON/OFF) from wearable accelerometer sensor data.

## Overview

This API analyzes accelerometer data uploaded as CSV files and returns:
- **Detected State**: `ON`, `OFF`, or `UNKNOWN`
- **Confidence Score**: 0.0 to 1.0
- **Explanation**: Human-readable analysis summary
- **Sensor Statistics**: Data quality metrics

## Project Structure

```
backend/
├── main.py           # FastAPI application & endpoints
├── predictor.py      # Prediction logic (rule-based placeholder)
├── utils.py          # CSV parsing & data validation
├── schemas.py        # Pydantic models for API contracts
├── config.py         # Centralized configuration
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Quick Start

### 1. Create Virtual Environment (Recommended)

```bash
cd backend
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message & API info |
| GET | `/health` | Health check & model status |
| GET | `/config` | Current configuration |
| POST | `/predict` | Upload CSV for state prediction |

## CSV File Format

Your CSV file should contain accelerometer data with the following columns:

| Column | Alternatives | Description |
|--------|--------------|-------------|
| `acc_x` | `x`, `accel_x`, `accelerometer_x` | X-axis acceleration |
| `acc_y` | `y`, `accel_y`, `accelerometer_y` | Y-axis acceleration |
| `acc_z` | `z`, `accel_z`, `accelerometer_z` | Z-axis acceleration |

### Requirements
- **Duration**: 5-10 seconds of data
- **Sampling Rate**: ~25Hz (125-320 samples)
- **Units**: m/s² (standard accelerometer units)

### Example CSV

```csv
acc_x,acc_y,acc_z
0.12,-0.34,9.75
0.15,-0.32,9.78
0.18,-0.30,9.81
...
```

## Testing

### Using Swagger UI

1. Navigate to http://localhost:8000/docs
2. Click on `POST /predict`
3. Click "Try it out"
4. Upload your CSV file
5. Click "Execute"

### Using cURL

```bash
# Basic prediction request
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_sensor_data.csv"

# Health check
curl http://localhost:8000/health

# Get configuration
curl http://localhost:8000/config
```

### Using Python

```python
import requests

# Upload file for prediction
with open("sensor_data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("sensor_data.csv", f, "text/csv")}
    )
    print(response.json())
```

### Using HTTPie

```bash
http POST localhost:8000/predict file@sensor_data.csv
```

## Sample Response

```json
{
  "filename": "patient_001_session.csv",
  "detected_state": "ON",
  "confidence": 0.82,
  "explanation": "Movement patterns indicate controlled motor function consistent with medication effectiveness (ON state). Key observations: Normal movement intensity observed. Smooth movement patterns detected. Consistent movement patterns.",
  "sensor_stats": {
    "sample_count": 250,
    "duration_seconds": 10.0,
    "mean_magnitude": 9.85,
    "std_magnitude": 0.45,
    "movement_intensity": 0.35
  }
}
```

## Create Test Data

Generate a sample CSV for testing:

```python
import numpy as np
import pandas as pd

# Simulate 10 seconds at 25Hz (ON state - low tremor)
np.random.seed(42)
n_samples = 250
t = np.linspace(0, 10, n_samples)

# Gravity on Z-axis + small noise (simulating watch worn on wrist)
data = pd.DataFrame({
    'acc_x': np.random.normal(0.1, 0.2, n_samples),
    'acc_y': np.random.normal(-0.2, 0.15, n_samples),
    'acc_z': np.random.normal(9.8, 0.3, n_samples)
})

data.to_csv('test_on_state.csv', index=False)
print("Created test_on_state.csv")

# Simulate OFF state - add tremor (4-6 Hz oscillation)
tremor_freq = 5  # Hz
tremor_amp = 0.8
data_off = pd.DataFrame({
    'acc_x': np.random.normal(0.1, 0.3, n_samples) + tremor_amp * np.sin(2 * np.pi * tremor_freq * t),
    'acc_y': np.random.normal(-0.2, 0.25, n_samples) + tremor_amp * 0.7 * np.sin(2 * np.pi * tremor_freq * t + 0.5),
    'acc_z': np.random.normal(9.8, 0.5, n_samples)
})

data_off.to_csv('test_off_state.csv', index=False)
print("Created test_off_state.csv")
```

## Configuration

Environment variables (prefix with `ACARE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ACARE_DEBUG` | `true` | Enable debug mode |
| `ACARE_EXPECTED_SAMPLE_RATE_HZ` | `25` | Expected sensor sampling rate |
| `ACARE_MIN_SAMPLES` | `125` | Minimum samples (~5 sec) |
| `ACARE_MAX_SAMPLES` | `320` | Maximum samples (~10 sec) |
| `ACARE_CONFIDENCE_THRESHOLD` | `0.6` | Below this returns UNKNOWN |

## Extending with ML Model

The codebase is designed for easy ML model integration:

1. **Edit `predictor.py`**:
   ```python
   class MLPredictor(BasePredictor):
       def __init__(self, model_path: str):
           import onnxruntime as ort
           self.session = ort.InferenceSession(model_path)
           self._ready = True
       
       def predict(self, sensor_stats: Dict[str, float]) -> PredictionResult:
           # Prepare features matching your training pipeline
           features = np.array([[
               sensor_stats['movement_intensity'],
               sensor_stats['avg_zcr'],
               sensor_stats['std_magnitude'],
               # ... other features
           ]], dtype=np.float32)
           
           # Run inference
           outputs = self.session.run(None, {'input': features})
           probabilities = outputs[0][0]
           
           # Map to state
           state_idx = np.argmax(probabilities)
           states = [PatientState.ON, PatientState.OFF, PatientState.UNKNOWN]
           
           return PredictionResult(
               state=states[state_idx],
               confidence=float(probabilities[state_idx]),
               explanation="ML model prediction",
               debug_info={'probabilities': probabilities.tolist()}
           )
   ```

2. **Update factory function** in `predictor.py`:
   ```python
   def get_predictor() -> BasePredictor:
       global _predictor_instance
       if _predictor_instance is None:
           _predictor_instance = MLPredictor("models/parkinson_detector.onnx")
       return _predictor_instance
   ```

## Error Handling

| HTTP Code | Error Type | Description |
|-----------|------------|-------------|
| 400 | `invalid_file_type` | File is not a CSV |
| 400 | `empty_file` | Uploaded file is empty |
| 422 | `data_validation_error` | Missing columns, insufficient data |
| 500 | `processing_error` | Unexpected server error |

## Development Notes

- **Current State**: Uses rule-based placeholder prediction
- **Production**: Replace with validated ML model (Random Forest / LSTM)
- **Validation**: Patient-wise cross-validation required before deployment

## License

Proprietary - aCare Health Technologies

---

For questions or issues, contact the aCare development team.
