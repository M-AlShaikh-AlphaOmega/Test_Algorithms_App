# RF_Test.py - Parkinson's ON/OFF Detection Pipeline

## What This Script Does

This script detects whether a Parkinson's patient is in an **ON state** (medication working) or **OFF state** (medication worn off) using accelerometer data from a wrist sensor.

It reads a raw binary sensor file (`Data_Jan-29`), decodes it, classifies the signal, and trains a Random Forest model to distinguish between the two states.

---

## How It Works (3 Main Steps)

### Step A: Decode the Binary Data

The sensor records movement as raw binary floats. The script loads `Data_Jan-29` (binary file) and `meta_Data_Jan-29.json` (metadata) using the decoder module (`Decode_SampleDataset_Jan-29.py`).

The decoder reads the binary file, unpacks the 32-bit floats, and produces a table with columns:

| time | X | Y | Z |
|------|------|------|------|
| 0.00 | -6.51 | -2.89 | 6.63 |
| 0.04 | -5.77 | -3.31 | 6.58 |
| ... | ... | ... | ... |

- **X, Y, Z** are accelerometer readings (3 axes of movement)
- **time** is in seconds, based on a 25 Hz sampling rate
- The current dataset has **91 samples** covering about **3.6 seconds**

### Step B: Classify the Signal into ON/OFF

Since the raw data has no labels, the script classifies each portion of the signal based on **tremor-band power analysis**:

1. The signal is split into **1-second sliding windows** with 50% overlap
2. For each window, it computes the **FFT** (frequency breakdown) of the acceleration magnitude
3. It measures how much energy falls in the **4-6 Hz tremor band**
4. Windows with tremor power **above the median** are labeled **OFF** (more shaking = medication worn off)
5. Windows with tremor power **below the median** are labeled **ON** (smoother motion = medication working)

This classification is based on the medical fact that Parkinson's tremor occurs primarily in the 4-6 Hz frequency range, and worsens during OFF states.

### Step C: Train the Random Forest Model

The labeled data is fed into a 9-step pipeline that trains and evaluates a Random Forest classifier.

---

## The 9-Step Pipeline

Once the data is decoded and classified, the `CompletePipeline` runs these steps:

### Step 1: Load Data
Receives the decoded and labeled DataFrame directly.

### Step 2: Preprocess
- **Gravity normalization** - scales accelerometer values relative to gravity (9.8 m/s^2)
- **Butterworth bandpass filter** - removes noise outside the 0.5-10 Hz range
- **Outlier removal** - replaces extreme values (beyond 5 standard deviations) with interpolated values

### Step 3: Extract Features
The signal is split into overlapping windows (1 second, 50% overlap). For each window, **52+ features** are extracted:

**Per axis (X, Y, Z) - 15 features each:**
- Time-domain: mean, std, variance, RMS, peak-to-peak, energy, SMA, skewness, kurtosis, IQR, zero-crossing rate
- Frequency-domain: dominant frequency, tremor-band power (4-6 Hz), voluntary-band power (0-3 Hz), spectral entropy

**Cross-axis - 7 features:**
- Magnitude mean, std, RMS
- Axis correlations (XY, XZ, YZ)
- Signal Vector Magnitude (SVM)

### Step 4: Learn Baseline
Calculates statistics (mean, std, median) from ON-state windows to establish the patient's personal "normal" movement baseline.

### Step 5: Compute Deviations
Adds z-score features measuring how much each window deviates from the patient's baseline. This roughly doubles the feature count.

### Step 6: Prepare Data
Filters out any unlabeled windows and encodes labels as numbers (OFF=0, ON=1).

### Step 7: Train Model
- Normalizes features using StandardScaler (mean=0, std=1)
- Auto-adjusts Random Forest parameters and CV folds for small datasets
- Runs stratified cross-validation
- Trains the final Random Forest model on all data

### Step 8: Evaluate
Calculates accuracy, precision, recall, F1-score, and confusion matrix on the training data.

### Step 9: Feature Importance
Ranks which features contribute most to the model's predictions.

---

## Output Files

After the pipeline completes, three files are saved:

| File | Location | Description |
|------|----------|-------------|
| `parkinsons_model.pkl` | `models/` | Trained model, scaler, feature names, baseline stats |
| `results.json` | `results/` | All metrics and configuration in JSON format |
| `results.md` | `results/` | Human-readable markdown report |

---

## Required Files

The script expects these files in the same folder:

| File | Purpose |
|------|---------|
| `Data_Jan-29` | Raw binary sensor data (32-bit floats, little-endian) |
| `meta_Data_Jan-29.json` | Metadata: sample count, frequency, data order, timestamp |
| `Decode_SampleDataset_Jan-29.py` | Decoder module that converts binary to DataFrame |

---

## Configuration

All parameters are set in the `SystemConfig` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 25 Hz | Sensor recording frequency |
| `window_duration` | 1.0 s | Length of each analysis window |
| `window_overlap` | 0.5 | 50% overlap between consecutive windows |
| `tremor_band_low` | 4.0 Hz | Start of Parkinson's tremor frequency range |
| `tremor_band_high` | 6.0 Hz | End of Parkinson's tremor frequency range |
| `n_cv_folds` | 5 | Cross-validation folds (auto-reduced for small data) |
| `max_missing_ratio` | 0.1 | Maximum allowed missing data per window (10%) |
| `outlier_std_threshold` | 5.0 | Standard deviations before a value is considered an outlier |

---

## Code Structure

The script is organized into these classes:

| Class | Lines | Role |
|-------|-------|------|
| `SystemConfig` | 61-94 | All configurable parameters |
| `DataLoader` | 101-220 | Loads data from recordings folder or contract sessions |
| `Preprocessor` | 227-291 | Normalizes, filters, and cleans sensor data |
| `FeatureExtractor` | 298-446 | Splits data into windows and extracts 52+ features |
| `BaselineLearner` | 453-515 | Learns patient-specific baseline from ON-state data |
| `QualityGate` | 522-557 | Checks for missing data, off-wrist, and sensor saturation |
| `Trainer` | 564-685 | Trains and evaluates the Random Forest model |
| `CompletePipeline` | 692-904 | Connects all classes into a single 9-step pipeline |
| `main()` | 911-1034 | Entry point: decode, classify, run pipeline, save outputs |

---

## How to Run

```
python RF_Test.py
```

Make sure `Data_Jan-29`, `meta_Data_Jan-29.json`, and `Decode_SampleDataset_Jan-29.py` are in the same folder.

---

## Current Limitations

- **Small dataset**: Data_Jan-29 contains only 91 samples (3.6 seconds), producing about 5 analysis windows. This is not enough for reliable model training.
- **Auto-classified labels**: Since Data_Jan-29 has no real ON/OFF labels, the script assigns them based on tremor-band power. For production use, real labeled recordings are needed.
- **Cross-validation adjustment**: With very few samples, the script automatically reduces CV folds and tree parameters to avoid errors, but results should not be considered clinically valid.

For meaningful results, collect longer recordings (minutes, not seconds) with known ON/OFF labels using the `recordings/` folder format (`sample_001_ON.csv`, `sample_002_OFF.csv`).

---

## Medical Background

Parkinson's patients take Levodopa medication to control motor symptoms. The medication effect fluctuates:

- **ON state**: Medication is active. Movement is smoother and more controlled.
- **OFF state**: Medication has worn off. Tremor increases, movement becomes slower and more erratic.

The 4-6 Hz tremor band is the key signal. Parkinson's resting tremor peaks in this range. By measuring how much energy the accelerometer signal has in this band, the model can distinguish between ON and OFF states.
