# Sample Dataset - Jan 28

Guide for decoding the Jan-28 accelerometer sample data.

## Files

| File | Description |
|------|-------------|
| `meta_sampleDataset.json` | Metadata describing the binary file |
| `SampleDataset` | Binary sensor data (no extension) |
| `Decode_SampleDataset.py` | Decoder script |
| `results.csv` | Pre-decoded CSV output |

## Dataset Info

| Property | Value |
|----------|-------|
| Samples | 1,025 |
| Sampling rate | 25 Hz |
| Duration | 41 seconds |
| Sensor | Accelerometer (X, Y, Z) |
| Binary size | 12,300 bytes (1025 x 3 x 4) |
| Units | g (1g = 9.8 m/s^2) |

## Metadata Fields

```json
{
  "sample_count": 1025,
  "freq": "25",
  "unix_timestamp": 1769568059.905249,
  "source": ["acc"],
  "data_order": "xyz"
}
```

- `sample_count` - Number of measurement samples
- `freq` - Sampling rate (25 Hz = 25 samples/second)
- `unix_timestamp` - Recording start time (2026-01-28 10:14:19)
- `source` - Sensor type (`"acc"` = accelerometer)
- `data_order` - Axis interleaving order in binary file

## Binary Format

- 32-bit floats, little-endian
- 4 bytes per value
- Layout: `[X0, Y0, Z0, X1, Y1, Z1, ...]`
- File has **no extension**

File size formula: `sample_count x 3 axes x 4 bytes = 12,300 bytes`

## Decoded Output

```
          X      Y      Z
time
0.00  0.635  0.210 -0.667
0.04  0.638  0.209 -0.695
0.08  0.626  0.210 -0.709
...
```

Value ranges: X: 0.443-1.493g, Y: 0.209-1.001g, Z: -0.780-0.443g

## Usage

### Command Line

```bash
python Decode_SampleDataset.py meta_sampleDataset.json
python Decode_SampleDataset.py meta_sampleDataset.json --output results.csv
```

### Python

```python
from Decode_SampleDataset import decode_sensor_file

df, meta = decode_sensor_file('meta_sampleDataset.json')
print(f"Loaded {len(df)} samples, duration: {len(df)/25:.1f}s")
```

## Important Notes

1. Binary file name = metadata file name without `meta_` prefix
2. Binary file has no extension
3. Both files must be in the same directory
4. File size must match exactly: `sample_count x 3 x 4` bytes
