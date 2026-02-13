# Decoded IMU Dataset Documentation

Documentation for the decoded accelerometer + gyroscope sensor data from Parkinson's patients.

## Overview

This dataset contains decoded IMU (Inertial Measurement Unit) sensor data from wrist-worn devices. Binary files from `data/EncryptedData/` are converted to CSV using `DecodeDataset.py`.

## Data Channels

Each CSV file contains 6 channels of motion data:

| Column | Type | Measurement | Unit |
|--------|------|-------------|------|
| Ax | Accelerometer | X-axis acceleration | m/s^2 |
| Ay | Accelerometer | Y-axis acceleration | m/s^2 |
| Az | Accelerometer | Z-axis acceleration | m/s^2 |
| Gx | Gyroscope | X-axis rotation | rad/s |
| Gy | Gyroscope | Y-axis rotation | rad/s |
| Gz | Gyroscope | Z-axis rotation | rad/s |

## Folder Structure

```
Sample_Dataset_Acc_Gyro/
├── DecodeDataset.py              # Decoder script
├── DecodeDataset_Doc.md          # This file
├── DecodedData/                  # Decoded CSVs (gitignored)
│   ├── <patient_id>/            # Encrypted patient identifier
│   │   ├── 2025_07_16_04_32.csv
│   │   └── ...
│   └── ...
├── DecodedData_Test/             # Test set (gitignored)
├── DecodeDataset_Train_Test/     # Training pipeline
├── models/                       # Trained models (gitignored)
└── results/                      # Training results
```

### CSV Format

```csv
Sample,Ax,Ay,Az,Gx,Gy,Gz
0,0.245,-0.123,9.807,0.015,-0.008,0.003
1,0.251,-0.119,9.812,0.014,-0.009,0.002
...
```

### File Naming

Files are named by timestamp: `YYYY_MM_DD_HH_MM.csv`

## Specifications

| Property | Value |
|----------|-------|
| Sampling rate | 32 Hz |
| Original format | Binary (float32, 4 bytes/value) |
| Decoded format | CSV |
| Patient IDs | Encrypted/anonymized |

### Typical Values

| Condition | Ax, Ay | Az | Gx, Gy, Gz |
|-----------|--------|-----|-------------|
| At rest | ~0 m/s^2 | ~9.81 m/s^2 | ~0 rad/s |
| Walking | -2 to +2 m/s^2 | 8 to 11 m/s^2 | -1 to +1 rad/s |

## Running the Decoder

```bash
python DecodeDataset.py
```

**Input**: `data/EncryptedData/` (binary files)
**Output**: `DecodedData/` (CSV files per patient)

Invalid files (corrupted, incomplete, <24 bytes) are automatically skipped.

## Usage

```python
import pandas as pd
import numpy as np

data = pd.read_csv('DecodedData/patient_id/2025_07_16_04_32.csv')

# Duration
duration = len(data) / 32  # seconds

# Acceleration magnitude
acc_mag = np.sqrt(data['Ax']**2 + data['Ay']**2 + data['Az']**2)

# Gyroscope magnitude
gyro_mag = np.sqrt(data['Gx']**2 + data['Gy']**2 + data['Gz']**2)
```

## Processing Recommendations

1. **Filtering**: Butterworth low-pass filter (cutoff: 10-20 Hz)
2. **Windowing**: 10-second windows for feature extraction
3. **Normalization**: Z-score normalization per patient
4. **Validation**: Split data by patient (not by time) to avoid leakage

## Pipeline

```
Raw Binary -> Decode -> CSV -> Feature Extraction -> ML Model -> State Detection
```

## Privacy

- Patient IDs are encrypted folder names
- Data is anonymized (no personal information)
- Handle according to HIPAA/GDPR regulations
