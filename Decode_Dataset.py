"""
Decode Dataset - Binary IMU Data Decoder

This script processes binary IMU data files from the rw_backup directory,
decodes them into physical units (m/s² for accelerometer, rad/s for gyroscope),
and saves the results as CSV files.

Usage:
    python Decode_Dataset.py

Requirements:
    - Place rw_backup/ folder in the same directory as this script
    - numpy must be installed

Binary Format:
    - Data stored as float32 (4 bytes per value)
    - 6 values per sample: Ax, Ay, Az, Gx, Gy, Gz
    - Sampling rate: 32 Hz

Conversion Factors:
    - Accelerometer: raw * 9.807 / 4096 (m/s²)
    - Gyroscope: raw * π / (32 * 180) (rad/s)

Output:
    decoded_csv/
    ├── <user_id_1>/
    │   ├── <timestamp>.csv
    │   └── ...
    └── <user_id_2>/
        └── ...

Author: ACare Project
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def decode_and_convert(file_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Decode a binary IMU file and convert to physical units.

    Args:
        file_path: Path to the binary IMU file

    Returns:
        Dictionary with keys 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz' containing
        numpy arrays of sensor values in physical units, or None if file is invalid.

    Example:
        >>> data = decode_and_convert(Path("rw_backup/user/timestamp"))
        >>> if data:
        ...     print(f"Samples: {len(data['Ax'])}")
    """
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()

        # Must be multiple of 4 bytes for float32
        if len(raw) < 24 or len(raw) % 4 != 0:
            return None

        data = np.frombuffer(raw, dtype=np.float32)

        # Must have at least 6 values (one complete sample)
        if len(data) < 6:
            return None

        # Truncate to complete samples
        num_samples = len(data) // 6
        data = data[:num_samples * 6]

        # Reshape and convert
        samples = data.reshape(-1, 6)
        result = {
            'Ax': samples[:, 0] * 9.807 / 4096,  # m/s²
            'Ay': samples[:, 1] * 9.807 / 4096,  # m/s²
            'Az': samples[:, 2] * 9.807 / 4096,  # m/s²
            'Gx': samples[:, 3] * 3.14159 / (32 * 180),  # rad/s
            'Gy': samples[:, 4] * 3.14159 / (32 * 180),  # rad/s
            'Gz': samples[:, 5] * 3.14159 / (32 * 180),  # rad/s
        }
        return result
    except Exception:
        return None


def main() -> None:
    """
    Main function to decode all binary IMU files in rw_backup directory.

    Processes all user directories and saves decoded CSV files to decoded_csv/.
    Prints summary statistics upon completion.
    """
    rw_backup = Path('rw_backup')
    output_dir = Path('decoded_csv')
    output_dir.mkdir(exist_ok=True)

    total_files = 0
    total_samples = 0
    skipped = 0

    for user_folder in sorted(rw_backup.iterdir()):
        if not user_folder.is_dir():
            continue

        user_output = output_dir / user_folder.name
        user_output.mkdir(exist_ok=True)

        for data_file in sorted(user_folder.iterdir()):
            if data_file.is_dir():
                continue

            data = decode_and_convert(data_file)
            if data is None:
                skipped += 1
                continue

            csv_file = user_output / f'{data_file.name}.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Sample', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
                for i in range(len(data['Ax'])):
                    writer.writerow([
                        i,
                        data['Ax'][i],
                        data['Ay'][i],
                        data['Az'][i],
                        data['Gx'][i],
                        data['Gy'][i],
                        data['Gz'][i]
                    ])

            total_files += 1
            total_samples += len(data['Ax'])

            if total_files % 500 == 0:
                print(f'Processed {total_files} files...')

    print(f'\nDone!')
    print(f'Decoded: {total_files} files')
    print(f'Skipped: {skipped} files (incomplete/invalid)')
    print(f'Total samples: {total_samples:,}')
    print(f'Output: {output_dir.absolute()}')


if __name__ == '__main__':
    main()
