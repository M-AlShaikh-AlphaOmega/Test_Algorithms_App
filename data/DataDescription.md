# Data Description

## EncryptedData/

Contains raw binary sensor data collected from Parkinson's patients via BLE-connected wrist-worn devices.

### Structure

```
EncryptedData/
├── <patient_id>/           # Encrypted/anonymized patient identifier
│   ├── <timestamp>         # Binary sensor file (no extension)
│   ├── meta_<timestamp>.json  # Metadata for the binary file
│   └── ...
└── ...
```

### Binary File Format

Each binary file contains IMU sensor readings stored as **32-bit little-endian floats**.

- **Accelerometer only**: 3 values per sample (X, Y, Z), interleaved as `[x0, y0, z0, x1, y1, z1, ...]`
- **Accelerometer + Gyroscope**: 6 values per sample (Ax, Ay, Az, Gx, Gy, Gz)
- File size = `sample_count * channels * 4` bytes

### Metadata Format

Each `meta_*.json` file describes its corresponding binary file:

```json
{
  "sample_count": 273,
  "freq": "25",
  "unix_timestamp": 1769656943.512549,
  "source": ["acc"],
  "data_order": "xyz"
}
```

| Field            | Description                                      |
|------------------|--------------------------------------------------|
| `sample_count`   | Number of measurement samples                    |
| `freq`           | Sampling rate in Hz (typically 25 or 32)         |
| `unix_timestamp` | Recording start time (seconds since epoch)       |
| `source`         | Sensor types: `["acc"]` or `["acc", "gyro"]`     |
| `data_order`     | Axis ordering in binary data (e.g., `"xyz"`)     |

### Decoding

Use the decoder scripts in `Sample_Dataset_Acc/` or `Sample_Dataset_Acc_Gyro/` to convert binary files to CSV:

```python
from Decode_SampleDataset import decode_sensor_file

df, meta = decode_sensor_file('meta_file.json')
```

### Privacy

Patient folder names are encrypted identifiers. No personal information is stored in the data files.
