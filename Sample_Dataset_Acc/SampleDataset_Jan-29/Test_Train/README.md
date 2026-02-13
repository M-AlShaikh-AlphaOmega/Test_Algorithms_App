# RF_Test.py - Accelerometer-Only ON/OFF Detection Pipeline

End-to-end pipeline that decodes binary sensor data, classifies windows using tremor-band analysis, and trains a Random Forest model.

## What It Does

1. **Decode** - Reads `Data_Jan-29` (binary) + `meta_Data_Jan-29.json` (metadata) into a DataFrame
2. **Classify** - Labels each 1-second window as ON/OFF based on 4-6 Hz tremor-band power
3. **Train** - Runs a 9-step pipeline: load, preprocess, extract features, learn baseline, compute z-score deviations, prepare data, train model, evaluate, rank features

## Required Files

| File | Purpose |
|------|---------|
| `Data_Jan-29` | Raw binary sensor data (32-bit floats) |
| `meta_Data_Jan-29.json` | Metadata (sample count, frequency, timestamp) |
| `Decode_SampleDataset_Jan-29.py` | Decoder module |

## How to Run

```bash
python RF_Test.py
```

All three files must be in the same directory.

## Pipeline Steps

### Preprocessing
- Gravity normalization (relative to 9.8 m/s^2)
- Butterworth bandpass filter (0.5-10 Hz)
- Outlier removal (>5 std replaced with interpolated values)

### Feature Extraction (per 1-second window, 50% overlap)

**Per axis (X, Y, Z) - 15 features each:**
- Time-domain: mean, std, variance, RMS, peak-to-peak, energy, SMA, skewness, kurtosis, IQR, zero-crossing rate
- Frequency-domain: dominant frequency, tremor-band power (4-6 Hz), voluntary-band power (0-3 Hz), spectral entropy

**Cross-axis - 7 features:**
- Magnitude mean, std, RMS
- Correlations (XY, XZ, YZ)
- Signal Vector Magnitude (SVM)

**Total: 52+ raw features**, roughly doubled after baseline z-score deviations.

### Baseline Learning
Computes mean/std/median from ON-state windows to establish the patient's personal movement baseline. Z-scores measure deviation from this baseline.

## Configuration (`SystemConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 25 Hz | Sensor recording frequency |
| `window_duration` | 1.0s | Analysis window length |
| `window_overlap` | 0.5 | 50% overlap between windows |
| `tremor_band_low` | 4.0 Hz | Tremor frequency range start |
| `tremor_band_high` | 6.0 Hz | Tremor frequency range end |
| `n_cv_folds` | 5 | Cross-validation folds (auto-reduced for small data) |

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `parkinsons_model.pkl` | `models/` | Trained model + scaler + feature names + baseline stats |
| `results.json` | `results/` | All metrics and configuration |
| `results.md` | `results/` | Human-readable report |

## Limitations

- **Small dataset**: Data_Jan-29 contains only 91 usable samples (~3.6 seconds), producing ~5 analysis windows. Not enough for reliable training.
- **Auto-classified labels**: Labels are assigned by tremor-band power, not clinical observation. Real labeled data is needed for production.
- **Auto-adjusted parameters**: CV folds and tree parameters are reduced for small datasets; results are not clinically valid.

For meaningful results, use longer recordings with known ON/OFF labels.
