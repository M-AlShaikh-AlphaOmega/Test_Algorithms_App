# Sample Dataset - Jan 29

Guide for decoding the Jan-29 accelerometer sample data.

## Files

| File | Description |
|------|-------------|
| `meta_Data_Jan-29.json` | Metadata describing the data |
| `Data_Jan-29` | Binary sensor data (no extension) |
| `raw.json` | Same data in JSON format (alternative) |
| `Decode_SampleDataset_Jan-29.py` | Decoder script (supports both binary and JSON) |

## Dataset Info

| Property | Value |
|----------|-------|
| Samples | 273 |
| Sampling rate | 25 Hz |
| Duration | 10.92 seconds |
| Sensor | Accelerometer (X, Y, Z) |
| Binary size | 3,276 bytes (273 x 3 x 4) |
| JSON values | 819 (273 x 3) |

## Metadata Fields

```json
{
  "sample_count": 273,
  "freq": "25",
  "unix_timestamp": 1769656943.512549,
  "source": ["acc"],
  "data_order": "xyz"
}
```

## Binary Format

Same as Jan-28: 32-bit little-endian floats, interleaved `[X, Y, Z, X, Y, Z, ...]`, no file extension.

## JSON Alternative

`raw.json` contains the same data as a flat JSON array of floats:
```json
[x0, y0, z0, x1, y1, z1, ...]
```
819 values total (273 samples x 3 axes).

## Decoded Output

```
          X       Y       Z
time
0.00  -6.512  -2.888   6.630
0.04  -5.768  -3.312   6.580
0.08  -5.677  -1.855   6.917
...
```

**Note on units**: The exact unit is unknown, but values are linear to physical acceleration. The signal shape is preserved; scale depends on hardware.

## Usage

### Command Line

```bash
# From binary
python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json

# From JSON
python Decode_SampleDataset_Jan-29.py --json raw.json --meta meta_Data_Jan-29.json

# Save CSV
python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json --output results.csv
```

### Python

```python
from Decode_SampleDataset_Jan29 import decode_sensor_file

# From binary
df, meta = decode_sensor_file('meta_Data_Jan-29.json')

# From JSON
df, meta = decode_sensor_file('meta_Data_Jan-29.json', json_filepath='raw.json')
```

## Comparison with Jan-28

| Property | Jan-28 | Jan-29 |
|----------|--------|--------|
| Samples | 1,025 | 273 |
| Duration | 41s | 10.92s |
| Binary size | 12,300 bytes | 3,276 bytes |
| Has JSON | No | Yes |
| Data range | ~0-1.5g | ~-28 to +11 (raw units) |

## Notes

1. Binary file name = metadata file name without `meta_` prefix
2. Binary file has no extension; both files must be co-located
3. Use `--json raw.json` if binary parsing fails
4. Each BLE packet covers 5-10 seconds. Longer recordings produce multiple file pairs
5. BLE uses ACL+ARQ for reliable transmission; disconnects create additional file pairs
