# DecodedData Documentation
## aCare Project - IMU Sensor Dataset for Parkinson's Disease Monitoring

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [What is This Data?](#what-is-this-data)
3. [Why We Use This Data](#why-we-use-this-data)
4. [Data Structure](#data-structure)
5. [Data Specifications](#data-specifications)
6. [How to Use This Data](#how-to-use-this-data)
7. [Applications](#applications)
8. [Running the Decoder Script](#running-the-decoder-script)
9. [Important Notes](#important-notes)

---

## 🎯 Overview

This dataset contains **decoded IMU (Inertial Measurement Unit) sensor data** collected from Parkinson's disease patients using wearable devices (smartwatches). The data has been converted from binary format to human-readable CSV files.

**Original Folder:** `data/raw/EncryptedData/` (binary files)  
**Decoded Folder:** `Decode_Dataset/DecodedData/` (CSV files)

---

## 📊 What is This Data?

### IMU Sensors
IMU stands for **Inertial Measurement Unit** - a device that measures:
- **Accelerometer**: Linear acceleration (movement speed changes)
- **Gyroscope**: Angular velocity (rotation speed)

### Data Captured
Each file contains **6 channels** of motion data:

| Channel | Type | Measurement | Unit | Description |
|---------|------|-------------|------|-------------|
| **Ax** | Accelerometer | X-axis acceleration | m/s² | Forward/backward movement |
| **Ay** | Accelerometer | Y-axis acceleration | m/s² | Left/right movement |
| **Az** | Accelerometer | Z-axis acceleration | m/s² | Up/down movement |
| **Gx** | Gyroscope | X-axis rotation | rad/s | Pitch (tilting forward/back) |
| **Gy** | Gyroscope | Y-axis rotation | rad/s | Roll (tilting left/right) |
| **Gz** | Gyroscope | Z-axis rotation | rad/s | Yaw (turning left/right) |

---

## 🎯 Why We Use This Data

### Clinical Purpose
Parkinson's disease causes **motor fluctuations** between:
- **ON state**: Good mobility, medications working well
- **OFF state**: Movement difficulties, medications wearing off

### Detection Goals
This IMU data helps us:

1. **Detect OFF States**
   - Identify when patient movement drops below 60% of baseline
   - Monitor changes in gait patterns (walking style)
   - Track tremor and bradykinesia (slowness of movement)

2. **Predict Future OFF Episodes**
   - Analyze movement trends over 60-minute windows
   - Forecast when next OFF state will occur
   - Provide early warnings for medication timing

3. **Analyze Gait Characteristics**
   - Stride length and velocity
   - Arm swing amplitude
   - Turn duration and velocity
   - Step regularity and symmetry

4. **Clinical Decision Support**
   - Optimize medication timing
   - Personalize treatment plans
   - Track disease progression
   - Provide activity recommendations

---

## 📁 Data Structure

### Folder Organization
```

Decode_Dataset/
    └── DecodedData/
        ├── akwn2jojna1/                              # Patient ID (encrypted)
        │   ├── 2025_07_16_04_32.csv                  # Timestamp: July 16, 2025, 04:32
        │   ├── 2025_07_16_04_37.csv
        │   ├── 2025_07_16_04_42.csv
        │   └── ...
        ├── oA9zb5Au-Ng3klQwjuY28P.../               # Another patient
        │   ├── 2025_07_16_03_15.csv
        │   └── ...
        ├── oA9zb5CDE7H5yBrng1lHfW.../               # Another patient
        │   └── ...
        └── ...
```

### File Naming Convention
- **Format**: `YYYY_MM_DD_HH_MM.csv`
- **Example**: `2025_07_16_04_32.csv`
  - Date: July 16, 2025
  - Time: 04:32 (4:32 AM)

### CSV File Structure
```csv
Sample,Ax,Ay,Az,Gx,Gy,Gz
0,0.245,-0.123,9.807,0.015,-0.008,0.003
1,0.251,-0.119,9.812,0.014,-0.009,0.002
2,0.248,-0.125,9.809,0.016,-0.007,0.004
...
```

**Columns:**
- **Sample**: Row number (0, 1, 2, ...)
- **Ax, Ay, Az**: Acceleration values (m/s²)
- **Gx, Gy, Gz**: Gyroscope values (rad/s)

---

## ⚙️ Data Specifications

### Technical Details
- **Sampling Rate**: 32 Hz (32 samples per second)
- **Original Format**: Binary (float32, 4 bytes per value)
- **Decoded Format**: CSV (text, human-readable)
- **Data Type**: Time-series sensor data

### Conversion Formulas
```python
# Accelerometer (raw → m/s²)
acceleration = raw_value * 9.807 / 4096

# Gyroscope (raw → rad/s)
angular_velocity = raw_value * π / (32 * 180)
```

### Typical Values

**At Rest (Standing Still):**
- Ax ≈ 0 m/s²
- Ay ≈ 0 m/s²
- Az ≈ 9.81 m/s² (gravity)
- Gx, Gy, Gz ≈ 0 rad/s

**During Walking:**
- Ax, Ay: -2 to +2 m/s²
- Az: 8 to 11 m/s²
- Gx, Gy, Gz: -1 to +1 rad/s

---

## 💻 How to Use This Data

### 1. Reading CSV Files (Python)
```python
import pandas as pd

# Load a single file
data = pd.read_csv('Decode_Dataset/DecodedData/patient_id/2025_07_16_04_32.csv')

# Access specific columns
ax_values = data['Ax'].values
ay_values = data['Ay'].values
az_values = data['Az'].values

print(f"Total samples: {len(data)}")
print(f"Duration: {len(data) / 32:.2f} seconds")
```

### 2. Basic Analysis
```python
import numpy as np

# Calculate magnitude of acceleration
acceleration_magnitude = np.sqrt(data['Ax']**2 + data['Ay']**2 + data['Az']**2)

# Calculate movement intensity
movement_intensity = np.std(acceleration_magnitude)
print(f"Movement intensity: {movement_intensity:.3f}")
```

### 3. Visualizing Data
```python
import matplotlib.pyplot as plt

# Create time axis (32 Hz sampling)
time = np.arange(len(data)) / 32  # Convert samples to seconds

# Plot acceleration
plt.figure(figsize=(12, 6))
plt.plot(time, data['Ax'], label='Ax')
plt.plot(time, data['Ay'], label='Ay')
plt.plot(time, data['Az'], label='Az')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s²)')
plt.title('3-Axis Acceleration Over Time')
plt.legend()
plt.grid(True)
plt.show()
```

### 4. Feature Extraction (for ML Models)
```python
# Extract features for OFF state detection
def extract_features(data):
    features = {}
    
    # Acceleration magnitude
    acc_mag = np.sqrt(data['Ax']**2 + data['Ay']**2 + data['Az']**2)
    
    # Statistical features
    features['mean_acc'] = np.mean(acc_mag)
    features['std_acc'] = np.std(acc_mag)
    features['max_acc'] = np.max(acc_mag)
    features['min_acc'] = np.min(acc_mag)
    
    # Gyroscope features
    gyro_mag = np.sqrt(data['Gx']**2 + data['Gy']**2 + data['Gz']**2)
    features['mean_gyro'] = np.mean(gyro_mag)
    features['std_gyro'] = np.std(gyro_mag)
    
    return features

features = extract_features(data)
print(features)
```

### 5. Processing Multiple Files
```python
from pathlib import Path

# Process all files for one patient
patient_folder = Path('Decode_Dataset/DecodedData/patient_id')

for csv_file in sorted(patient_folder.glob('*.csv')):
    data = pd.read_csv(csv_file)
    features = extract_features(data)
    print(f"{csv_file.name}: Movement = {features['mean_acc']:.3f} m/s²")
```

---

## 🚀 Applications

### 1. OFF State Detection
**Goal**: Identify when patient is in OFF state (current moment)

**Method**:
- Extract 130+ features from IMU data
- Use Random Forest classifier
- Compare with patient's baseline movement
- Detect when movement drops below 60% threshold

**Expected Accuracy**: 93-96%

### 2. OFF State Prediction
**Goal**: Forecast future OFF episodes (30-60 minutes ahead)

**Method**:
- Use LSTM neural networks
- Analyze 60-minute time windows
- Track movement trend decline
- Predict time until next OFF episode

**Use Case**: Proactive medication reminders

### 3. Gait Analysis
**Features Extracted**:
- Stride length and velocity
- Stride time and variability
- Arm swing amplitude
- Turn duration and velocity
- Step regularity and symmetry
- Cadence (steps per minute)

### 4. Activity Recognition
**Activities Detected**:
- Walking
- Standing
- Sitting
- Lying down
- Turning
- Going up/down stairs

---

## ⚠️ Important Notes

### Data Quality
✅ **Good Quality Files**:
- Complete samples (multiples of 6 values)
- Proper file size (≥ 24 bytes)
- Valid float32 format

❌ **Invalid Files** (automatically skipped):
- Corrupted binary data
- Incomplete samples
- Too small (<24 bytes)

### Privacy & Security
- Patient IDs are **encrypted** (folder names)
- Data is **anonymized** (no personal information)
- Follow **HIPAA/GDPR** regulations when handling

### Processing Recommendations
1. **Preprocessing**: Apply Butterworth low-pass filter (cutoff: 10-20 Hz)
2. **Windowing**: Use 10-second windows for feature extraction
3. **Normalization**: Z-score normalization per patient
4. **Validation**: Split data by patient (not by time) to avoid leakage

### Next Steps in Pipeline
```
Raw Binary → Decode → CSV Files → Feature Extraction → ML Models → Clinical Insights
                     (You are here)
```

**After decoding**, you should:
1. Extract features (gait parameters, statistical features)
2. Train machine learning models (Random Forest, LSTM)
3. Validate with clinical data (MDS-UPDRS scores)
4. Deploy to real-time monitoring system

---

## 🔧 Running the Decoder Script

### Project Structure
Your project should be organized like this:
```
your_project/
├── data/
│   ├── raw/
│   │   └── EncryptedData/           # Your encrypted binary files
│   │       ├── akwn2jojna1/
│   │       ├── oA9zb5Au-Ng3klQwjuY28P.../
│   │       ├── oA9zb5CDE7H5yBrng1lHfW.../
│   │       └── ...
│   ├── interim/
│   ├── processed/
|
│── Decode_Dataset/
│       ├── DecodeDataset.py        # Decoder script
│       ├── DecodedDataset_Doc.md    # This documentation
│       └── DecodedData/             # Output folder (created automatically)
│           ├── akwn2jojna1/
│           │   ├── file1.csv
│           │   └── ...
│           └── ...
└── ...
```

### Steps to Decode Your Data

**Step 1**: Navigate to project root directory
```bash
cd your_project
```

**Step 2**: Run the decoder script
```bash
python Decode_Dataset/DecodeDataset.py
```

**Alternative**: Navigate to Decode_Dataset folder first
```bash
cd Decode_Dataset
python DecodeDataset.py
```

**Step 3**: Check the output
- Decoded CSV files will be in: `data/Decode_Dataset/DecodedData/`
- Each patient folder will contain CSV files for all their recordings

### Expected Output
```
Processing started...
Input: /path/to/your_project/data/raw/EncryptedData
Output: /path/to/your_project/data/Decode_Dataset/DecodedData

Processed 500 files...
Processed 1000 files...
...

==================================================
Decoding Complete!
==================================================
✓ Successfully decoded: 1234 files
✗ Skipped (invalid): 5 files
📊 Total samples: 15,678,900
📁 Output directory: /path/to/your_project/data/Decode_Dataset/DecodedData
==================================================
```

---

## 📚 Additional Resources

### Related Project Files
- `Decode_Dataset.py`: Script that generated this data
- Algorithm documentation: See project's ALGO cards
- Feature extraction: See SCHEMA cards
- Clinical validation: See PILOT documentation

### Contact & Support
For questions about this dataset or the aCare project, refer to project documentation or contact the development team.

---

**Last Updated**: January 2026  
**Project**: aCare - Parkinson's Disease Monitoring System  
**Version**: 1.0