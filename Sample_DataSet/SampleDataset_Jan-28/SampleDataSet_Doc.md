# aCare Sample Dataset - README

Guide for decoding aCare sensor data files.

---

## Files Overview

### 1. meta_sampleDataset.json (Metadata)
Contains information about the data structure.

**Fields:**
- `sample_count`: Number of measurements (1025)
- `freq`: Sampling rate in Hz (25 = 25 samples/second)
- `unix_timestamp`: Recording start time
- `source`: Sensor type (["acc"] = accelerometer)
- `data_order`: Axis order ("xyz" = X, Y, Z)

**Why needed:** Tells decoder how to read the binary file correctly.

---

### 2. SampleDataset (Binary Data)
Contains actual sensor measurements.

**Format:**
- 32-bit floats, little-endian
- 4 bytes per value
- Size: 12,300 bytes (1025 × 3 × 4)
- **No extension** (not .bin or .dat)

**Why binary:** Saves space and improves speed.

---

### 3. Decode_SampleDataset.py (Decoder)
Converts binary data to readable format (CSV or DataFrame).

---

## Understanding Metadata

### sample_count (1025)
Total number of measurements in the file.

**Used for:**
- Calculate expected file size
- Validate data integrity
- Determine output rows

---

### freq ("25")
Sampling rate = 25 measurements per second.

**Calculations:**
```
Time between samples = 1/25 = 0.04 seconds
Total duration = 1025/25 = 41 seconds
```

---

### unix_timestamp
Exact recording start time (seconds since Jan 1, 1970).

**Example:**
```python
import datetime
timestamp = 1769568059.905249
dt = datetime.datetime.fromtimestamp(timestamp)
# Result: 2026-01-28 10:14:19
```

---

### source (["acc"])
Sensor types used.

**Options:**
- `"acc"` = Accelerometer (movement/vibration)
- `"gyro"` = Gyroscope (rotation)

---

### data_order ("xyz")
Order of axes in binary file.

**Critical:** Must read in correct order.

**Layout:**
```
Sample 0: X, Y, Z
Sample 1: X, Y, Z
Sample 2: X, Y, Z
...
```

---

## Decoded Data Structure

After decoding:

```
          X      Y      Z
time                     
0.00  0.635  0.210 -0.667
0.04  0.638  0.209 -0.695
0.08  0.626  0.210 -0.709
...
```

**Columns:**
- `time`: Seconds from start (0.00 to 40.96)
- `X`: Horizontal acceleration (g units)
- `Y`: Vertical acceleration (g units)
- `Z`: Forward/backward acceleration (g units)

**Note:** 1g = 9.8 m/s² (Earth's gravity)

---

## How to Use

### Prerequisites
```bash
pip install pandas numpy
```

### Command Line

**Basic:**
```bash
python Decode_SampleDataset.py meta_sampleDataset.json
```

**Save CSV:**
```bash
python Decode_SampleDataset.py meta_sampleDataset.json --output results.csv
```

### Python Code

**Basic:**
```python
from Decode_SampleDataset import decode_sensor_file

df, meta = decode_sensor_file('meta_sampleDataset.json')
print(f"Loaded {len(df)} samples")
```

**Silent mode:**
```python
df, meta = decode_sensor_file('meta_sampleDataset.json', verbose=False)
```

**With CSV:**
```python
df, meta = decode_sensor_file('meta_sampleDataset.json', output_csv='data.csv')
```

---

## Analysis Examples

### Example 1: Calculate Magnitude
```python
import numpy as np
from Decode_SampleDataset import decode_sensor_file

df, meta = decode_sensor_file('meta_sampleDataset.json', verbose=False)

# Total acceleration magnitude
magnitude = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

print(f"Mean: {magnitude.mean():.3f}g")
print(f"Max: {magnitude.max():.3f}g")
```

### Example 2: Time Window
```python
from Decode_SampleDataset import decode_sensor_file

df, meta = decode_sensor_file('meta_sampleDataset.json', verbose=False)

# Get data from 10 to 20 seconds
window = df.loc[10.0:20.0]

print(f"Samples in window: {len(window)}")
print(f"Mean X: {window['X'].mean():.3f}g")
```

### Example 3: Plot Data
```python
import matplotlib.pyplot as plt
from Decode_SampleDataset import decode_sensor_file

df, meta = decode_sensor_file('meta_sampleDataset.json', verbose=False)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['X'], label='X')
plt.plot(df.index, df['Y'], label='Y')
plt.plot(df.index, df['Z'], label='Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Troubleshooting

### Error: File not found
**Problem:** `FileNotFoundError: meta_sampleDataset.json not found`

**Solution:**
- Check file exists in current directory
- Verify filename spelling
- Use full path if needed

### Error: Binary file not found
**Problem:** `FileNotFoundError: Binary file not found: SampleDataset`

**Solution:**
- Both files must be in same directory
- Binary file should have NO extension
- Check filename matches (meta_X.json → X)

### Error: File size mismatch
**Problem:** `ValueError: File size mismatch: expected 12300, got ...`

**Solution:**
- File may be corrupted
- Re-download or re-copy file
- Check transfer completed

### Error: Missing pandas
**Problem:** `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
pip install pandas numpy
```

---

## Quick Reference

### Sample Data Info
```
Samples: 1,025
Frequency: 25 Hz
Duration: 41 seconds
File size: 12,300 bytes
Sensor: Accelerometer
Axes: X, Y, Z

Value ranges:
  X: 0.443 to 1.493g
  Y: 0.209 to 1.001g
  Z: -0.780 to 0.443g
```

### File Size Formula
```
sample_count × num_axes × bytes_per_float
= 1025 × 3 × 4 = 12,300 bytes
```

### Duration Formula
```
duration = sample_count / frequency
        = 1025 / 25 = 41 seconds
```

---

## Important Notes

1. **File naming:** Binary file = metadata file without "meta_" prefix
2. **File format:** 32-bit little-endian floats
3. **No extension:** Binary file has NO file extension
4. **Same directory:** Both files must be in same location
5. **Exact size:** File size must match: sample_count × 3 × 4 bytes

---

## Summary

**What you have:**
- Metadata file (JSON) - describes structure
- Binary data file - contains measurements
- Decoder script (Python) - converts binary to readable

**What you can do:**
- Decode binary to CSV/DataFrame
- Analyze movement patterns
- Extract features for ML models
- Visualize acceleration data

**Key point:** Metadata tells decoder HOW to read binary file. Both files required.