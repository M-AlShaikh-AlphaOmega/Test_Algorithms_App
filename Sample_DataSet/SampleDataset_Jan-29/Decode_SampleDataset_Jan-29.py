"""
aCare Sample Dataset Decoder (Jan-29)
=====================================
Decodes sensor data files for Parkinson's disease monitoring.
Supports both binary and JSON formats.

Usage:
    python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json
    python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json --output results.csv
    python Decode_SampleDataset_Jan-29.py --json raw.json --meta meta_Data_Jan-29.json
"""

import struct
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_metadata(meta_filepath: str) -> Dict:
    """
    Load metadata from JSON file.

    Args:
        meta_filepath: Path to meta JSON file

    Returns:
        Dictionary with metadata (sample_count, freq, source, data_order, timestamp)
    """
    try:
        with open(meta_filepath, 'r') as f:
            meta = json.load(f)
        return meta
    except FileNotFoundError:
        raise FileNotFoundError(f"Meta file not found: {meta_filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def validate_metadata(meta: Dict) -> None:
    """
    Check if metadata has all required fields and valid values.

    Required fields: sample_count, freq, data_order, source, unix_timestamp
    """
    required = ['sample_count', 'freq', 'data_order', 'source', 'unix_timestamp']
    missing = [f for f in required if f not in meta]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if int(meta['sample_count']) <= 0:
        raise ValueError(f"Invalid sample_count: {meta['sample_count']}")

    if float(meta['freq']) <= 0:
        raise ValueError(f"Invalid frequency: {meta['freq']}")


def get_binary_filepath(meta_filepath: str) -> Path:
    """
    Find binary data file based on meta filename.
    Example: meta_Data_Jan-29.json -> Data_Jan-29

    Returns:
        Path to binary file (no extension)
    """
    meta_path = Path(meta_filepath)

    # Remove 'meta_' prefix to get data filename
    meta_name = meta_path.stem  # e.g., 'meta_Data_Jan-29'
    if meta_name.startswith('meta_'):
        data_name = meta_name.replace('meta_', '')
    else:
        data_name = meta_name

    data_filepath = meta_path.parent / data_name

    if not data_filepath.exists():
        raise FileNotFoundError(f"Binary file not found: {data_filepath}")

    return data_filepath


def validate_file_size(binary_data: bytes, meta: Dict) -> None:
    """
    Verify binary file size matches expected size.
    Expected = sample_count × num_axes × num_sensors × 4 bytes
    """
    sample_count = int(meta['sample_count'])
    num_axes = len(meta['data_order'])
    num_sensors = len(meta['source'])

    expected_bytes = sample_count * num_axes * num_sensors * 4
    actual_bytes = len(binary_data)

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )


def load_json_data(json_filepath: str) -> list:
    """
    Load sensor data from JSON array file.

    Args:
        json_filepath: Path to JSON file containing float array

    Returns:
        List of float values
    """
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of floats")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def decode_binary_data(binary_data: bytes, meta: Dict) -> pd.DataFrame:
    """
    Decode binary sensor data into pandas DataFrame.

    Binary format:
        - 32-bit float (4 bytes per value)
        - Little-endian byte order
        - Sample-by-sample, axis-by-axis layout

    Args:
        binary_data: Raw binary data
        meta: Metadata dictionary

    Returns:
        DataFrame with columns X, Y, Z (and time index)
    """
    sample_count = int(meta['sample_count'])
    data_order = meta['data_order']
    source = meta['source']
    freq = float(meta['freq'])

    num_axes = len(data_order)
    num_sensors = len(source)
    total_floats = sample_count * num_axes * num_sensors

    # Unpack binary as little-endian floats
    floats = struct.unpack(f'<{total_floats}f', binary_data)

    return _floats_to_dataframe(floats, meta)


def decode_json_data(json_data: list, meta: Dict) -> pd.DataFrame:
    """
    Convert JSON float array to pandas DataFrame.

    Args:
        json_data: List of float values
        meta: Metadata dictionary

    Returns:
        DataFrame with columns X, Y, Z (and time index)
    """
    return _floats_to_dataframe(json_data, meta)


def _floats_to_dataframe(floats: list, meta: Dict) -> pd.DataFrame:
    """
    Convert float array to pandas DataFrame.

    Args:
        floats: List or tuple of float values
        meta: Metadata dictionary

    Returns:
        DataFrame with columns X, Y, Z (and time index)
    """
    sample_count = int(meta['sample_count'])
    data_order = meta['data_order']
    source = meta['source']
    freq = float(meta['freq'])

    num_axes = len(data_order)
    num_sensors = len(source)

    # Organize into DataFrame
    if num_sensors == 1:
        # Single sensor (e.g., only accelerometer)
        data = {axis.upper(): [] for axis in data_order}
        for i in range(sample_count):
            for j, axis in enumerate(data_order):
                idx = i * num_axes + j
                data[axis.upper()].append(floats[idx])
    else:
        # Multiple sensors (e.g., accelerometer + gyroscope)
        data = {}
        for sensor in source:
            for axis in data_order:
                col_name = f"{sensor}_{axis.upper()}"
                data[col_name] = []

        for i in range(sample_count):
            for sensor_idx, sensor in enumerate(source):
                for j, axis in enumerate(data_order):
                    idx = i * num_axes * num_sensors + sensor_idx * num_axes + j
                    col_name = f"{sensor}_{axis.upper()}"
                    data[col_name].append(floats[idx])

    # Create DataFrame with time index
    df = pd.DataFrame(data)
    df['time'] = np.arange(sample_count) / freq
    df = df.set_index('time')

    return df


def decode_sensor_file(meta_filepath: str, output_csv: Optional[str] = None,
                       verbose: bool = True, json_filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to decode aCare sensor file.

    Process:
        1. Load and validate metadata
        2. Read binary or JSON data file
        3. Validate data size
        4. Decode to DataFrame
        5. Optionally save to CSV

    Args:
        meta_filepath: Path to metadata JSON file
        output_csv: Optional path to save CSV output
        verbose: Print progress messages if True
        json_filepath: Optional path to JSON data file (if provided, uses JSON instead of binary)

    Returns:
        Tuple of (DataFrame with sensor data, metadata dict)

    Example:
        >>> df, meta = decode_sensor_file('meta_Data_Jan-29.json')
        >>> df, meta = decode_sensor_file('meta_Data_Jan-29.json', 'output.csv')
        >>> df, meta = decode_sensor_file('meta_Data_Jan-29.json', json_filepath='raw.json')
    """
    try:
        if verbose:
            print(f"\n{'='*60}")
            print("aCare Sensor Data Decoder (Jan-29)")
            print(f"{'='*60}\n")

        # Step 1: Load metadata
        if verbose:
            print(f"[1/4] Loading metadata...")
        meta = load_metadata(meta_filepath)
        validate_metadata(meta)

        # Print key info
        if verbose:
            print(f"      Samples: {meta['sample_count']}")
            print(f"      Frequency: {meta['freq']} Hz")
            print(f"      Sensor: {meta['source'][0]}")
            print(f"      Duration: {int(meta['sample_count']) / float(meta['freq']):.2f}s")

        # Step 2: Read data file (binary or JSON)
        if json_filepath:
            if verbose:
                print(f"\n[2/4] Reading JSON file...")
            json_data = load_json_data(json_filepath)

            if verbose:
                json_path = Path(json_filepath)
                print(f"      File: {json_path.name}")
                print(f"      Values: {len(json_data):,}")

            # Step 3: Validate
            if verbose:
                print(f"\n[3/4] Validating...")
            expected_values = int(meta['sample_count']) * len(meta['data_order']) * len(meta['source'])
            if len(json_data) != expected_values:
                raise ValueError(f"Data size mismatch: expected {expected_values}, got {len(json_data)}")

            if verbose:
                print(f"      Data count: OK")

            # Step 4: Decode
            if verbose:
                print(f"\n[4/4] Decoding...")
            df = decode_json_data(json_data, meta)
        else:
            if verbose:
                print(f"\n[2/4] Reading binary file...")
            data_filepath = get_binary_filepath(meta_filepath)
            with open(data_filepath, 'rb') as f:
                binary_data = f.read()

            if verbose:
                print(f"      File: {data_filepath.name}")
                print(f"      Size: {len(binary_data):,} bytes")

            # Step 3: Validate
            if verbose:
                print(f"\n[3/4] Validating...")
            validate_file_size(binary_data, meta)

            if verbose:
                print(f"      File size: OK")

            # Step 4: Decode
            if verbose:
                print(f"\n[4/4] Decoding...")
            df = decode_binary_data(binary_data, meta)

        if verbose:
            print(f"      Shape: {df.shape}")
            print(f"      Columns: {list(df.columns)}")

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv)
            if verbose:
                print(f"\n      Saved to: {output_csv}")

        # Show summary
        if verbose:
            print(f"\n{'='*60}")
            print("Summary Statistics:")
            print(f"{'='*60}")
            print(df.describe())
            print(f"\n{'='*60}")
            print("Decoding completed successfully")
            print(f"{'='*60}\n")

        return df, meta

    except Exception as e:
        print(f"\nError: {e}\n")
        raise


# Command line interface
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
Usage:
    python Decode_SampleDataset_Jan-29.py <meta_file.json> [--output <file.csv>]
    python Decode_SampleDataset_Jan-29.py --json <raw.json> --meta <meta_file.json> [--output <file.csv>]

Examples:
    python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json
    python Decode_SampleDataset_Jan-29.py meta_Data_Jan-29.json --output results.csv
    python Decode_SampleDataset_Jan-29.py --json raw.json --meta meta_Data_Jan-29.json

Required files:
    - meta_Data_Jan-29.json  (metadata)
    - Data_Jan-29            (binary data) OR raw.json (JSON data)
        """)
        sys.exit(1)

    meta_file = None
    json_file = None
    output_csv = None

    # Parse arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--output':
            if i + 1 < len(sys.argv):
                output_csv = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --output requires a filename")
                sys.exit(1)
        elif arg == '--json':
            if i + 1 < len(sys.argv):
                json_file = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --json requires a filename")
                sys.exit(1)
        elif arg == '--meta':
            if i + 1 < len(sys.argv):
                meta_file = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --meta requires a filename")
                sys.exit(1)
        else:
            # Assume it's the meta file if not specified
            if meta_file is None:
                meta_file = arg
            i += 1

    if meta_file is None:
        print("Error: metadata file is required")
        sys.exit(1)

    # Decode file
    df, meta = decode_sensor_file(meta_file, output_csv, json_filepath=json_file)
