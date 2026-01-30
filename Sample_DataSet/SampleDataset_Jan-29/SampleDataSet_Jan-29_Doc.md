# aCare Sample Dataset (Jan-29) - README

Guide for decoding aCare sensor data files.

---

## Files Overview

### 1. meta_Data_Jan-29.json (Metadata)
Contains information about the data structure.

**Fields:**
- `sample_count`: Number of measurements (273)
- `freq`: Sampling rate in Hz (25 = 25 samples/second)
- `unix_timestamp`: Recording start time (1769656943.512549)
- `source`: Sensor type (["acc"] = accelerometer)
- `data_order`: Axis order ("xyz" = X, Y, Z)

**Why needed:** Tells decoder how to read the data file correctly.

---

### 2. 1769656943 (Binary Data)
Contains actual sensor measurements in binary format.

**Format:**
- 32-bit floats, little-endian
- 4 bytes per value
- Size: 3,276 bytes (273 × 3 × 4)
- **No extension** (not .bin or .dat)

**Why binary:** Saves space and improves speed.

---

### 3. raw.json (JSON Data - Alternative Format)
Contains the same sensor data as JSON array.

**Format:**
- JSON array of float values
- 819 values total (273 samples × 3 axes)
- Interleaved: [x1, y1, z1, x2, y2, z2, ...]

**Why JSON:** Human-readable, easy debugging, no binary parsing needed.

---

### 4. Decode_SampleDataset_Jan-29.py (Decoder)
Converts binary or JSON data to readable format (CSV or DataFrame).

---

## Understanding Metadata

### sample_count (273)
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
Total duration = 273/25 = 10.92 seconds
```

---

### unix_timestamp
Exact recording start time (seconds since 00:00:00 Jan 1, 1970 at UTC+0).
It's globally unique and convertible to any time format.

**Example:**
```python
import datetime
timestamp = 1769656943.512549
dt = datetime.datetime.utcfromtimestamp(timestamp)
# Result: 2026-01-29 00:55:43 UTC
```

---

### source (["acc"])
Sensor types used.

**Options:**
- `"acc"` = Accelerometer (movement/vibration)
- `"gyro"` = Gyroscope (rotation)

---

### data_order ("xyz")
Order of axes in data file.

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
          X       Y       Z
time
0.00  -6.512  -2.888   6.630
0.04  -5.768  -3.312   6.580
0.08  -5.677  -1.855   6.917
...
```

**Columns:**
- `time`: Seconds from start (0.00 to 10.88)
- `X`: Horizontal acceleration
- `Y`: Vertical acceleration
- `Z`: Forward/backward acceleration

**Note on Units:** The exact data unit is unknown, but values are **linear to physical acceleration** with a constant scale. The signal shape is preserved - scale depends on hardware magnifier, digitizer, etc. Sensor manufacturers ensure linearity of sensor output.

---

## How to Use

### Prerequisites
```bash
pip install pandas numpy
```

### Command Line

**Basic (from binary):**
```bash
python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json
```

**Save CSV:**
```bash
python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json --output results.csv
```

**From JSON:**
```bash
python Decode_SampleDataset_Jan-29.py --json raw.json --meta meta_Data_Jan-29.json
```

### Python Code

**Basic:**
```python
from Decode_SampleDataset_Jan-29 import decode_sensor_file

df, meta = decode_sensor_file('meta_Data_Jan-29.json')
print(f"Loaded {len(df)} samples")
```

**Silent mode:**
```python
df, meta = decode_sensor_file('meta_Data_Jan-29.json', verbose=False)
```

**With CSV:**
```python
df, meta = decode_sensor_file('meta_Data_Jan-29.json', output_csv='data.csv')
```

**From JSON:**
```python
df, meta = decode_sensor_file('meta_Data_Jan-29.json', json_filepath='raw.json')
```

---

## Analysis Examples

### Example 1: Calculate Magnitude
```python
import numpy as np
from Decode_SampleDataset_Jan-29 import decode_sensor_file

df, meta = decode_sensor_file('meta_Data_Jan-29.json', verbose=False)

# Total acceleration magnitude
magnitude = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

print(f"Mean: {magnitude.mean():.3f} units")
print(f"Max: {magnitude.max():.3f} units")
```

### Example 2: Time Window
```python
from Decode_SampleDataset_Jan-29 import decode_sensor_file

df, meta = decode_sensor_file('meta_Data_Jan-29.json', verbose=False)

# Get data from 2 to 5 seconds
window = df.loc[2.0:5.0]

print(f"Samples in window: {len(window)}")
print(f"Mean X: {window['X'].mean():.3f} units")
```

### Example 3: Plot Data
```python
import matplotlib.pyplot as plt
from Decode_SampleDataset_Jan-29 import decode_sensor_file

df, meta = decode_sensor_file('meta_Data_Jan-29.json', verbose=False)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['X'], label='X')
plt.plot(df.index, df['Y'], label='Y')
plt.plot(df.index, df['Z'], label='Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (units)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Troubleshooting

### Error: File not found
**Problem:** `FileNotFoundError: meta_Data_Jan-29.json not found`

**Solution:**
- Check file exists in current directory
- Verify filename spelling
- Use full path if needed

### Error: Binary file not found
**Problem:** `FileNotFoundError: Binary file not found: Data_Jan-29`

**Solution:**
- Both files must be in same directory
- Binary file should have NO extension
- Check filename matches (meta_X.json -> X)
- Alternatively, use --json flag with raw.json

### Error: File size mismatch
**Problem:** `ValueError: File size mismatch: expected 3276, got ...`

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
Samples: 273
Frequency: 25 Hz
Duration: 10.92 seconds
Binary file size: 3,276 bytes
JSON values: 819
Sensor: Accelerometer
Axes: X, Y, Z
```

### File Size Formula (Binary)
```
sample_count × num_axes × bytes_per_float
= 273 × 3 × 4 = 3,276 bytes
```

### JSON Value Count Formula
```
sample_count × num_axes
= 273 × 3 = 819 values
```

### Duration Formula
```
duration = sample_count / frequency
        = 273 / 25 = 10.92 seconds
```

---

## Comparison with Jan-28 Dataset

| Property | Jan-28 | Jan-29 |
|----------|--------|--------|
| Samples | 1,025 | 273 |
| Duration | 41s | 10.92s |
| Binary Size | 12,300 bytes | 3,276 bytes |
| Has JSON | No | Yes (raw.json) |
| Data Range | ~0-1.5g | ~-28 to +11 (raw) |

---

## Important Notes

1. **File naming:** Binary file = metadata file without "meta_" prefix
2. **File format:** 32-bit little-endian floats
3. **No extension:** Binary file has NO file extension
4. **Same directory:** Both files must be in same location
5. **Exact size:** File size must match: sample_count × 3 × 4 bytes
6. **JSON alternative:** Use raw.json with --json flag if binary parsing fails
7. **Packet duration:** Each packet is 5-10 seconds. Longer recordings = more file pairs (meta+binary)
8. **Disconnection handling:** If BLE disconnects and retries, additional file pairs are created. BLE uses ACL+ARQ (similar to TCP) for reliable transmission
9. **Data units:** Unknown exact unit, but linear to physical acceleration (signal shape preserved)

---

## Summary

**What you have:**
- Metadata file (JSON) - describes structure
- Binary data file - contains measurements (compact)
- JSON data file - contains measurements (readable)
- Decoder script (Python) - converts to readable format

**What you can do:**
- Decode binary or JSON to CSV/DataFrame
- Analyze movement patterns
- Extract features for ML models
- Visualize acceleration data

**Key point:** Metadata tells decoder HOW to read data file. Both metadata and data files are required.
