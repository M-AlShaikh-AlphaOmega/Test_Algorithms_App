"""
Decode Dataset - Binary IMU Data Decoder for aCare Project

This script decodes binary IMU sensor files from data/raw/EncryptedData folder
and converts them to readable CSV files with physical units.

Usage:
    python DecodeDataset.py
    (Run from project root directory)

Requirements:
    - data/raw/EncryptedData/ folder with binary files
    - numpy installed (pip install numpy)

Author: aCare Project Team
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def decode_and_convert(file_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Decode binary IMU file and convert to physical units.
    
    Binary Format: float32 (4 bytes per value), 6 values per sample
    Sampling Rate: 32 Hz
    
    Args:
        file_path: Path to binary IMU file
        
    Returns:
        Dictionary with sensor data arrays or None if file is invalid
        Keys: 'Ax', 'Ay', 'Az' (m/s²) and 'Gx', 'Gy', 'Gz' (rad/s)
    """
    try:
        # Read binary file
        with open(file_path, 'rb') as f:
            raw = f.read()
        
        # Validate file size (must be multiple of 4 bytes for float32)
        if len(raw) < 24 or len(raw) % 4 != 0:
            return None
        
        # Convert binary data to float32 array
        data = np.frombuffer(raw, dtype=np.float32)
        
        # Must have at least 6 values (one complete sample)
        if len(data) < 6:
            return None
        
        # Ensure we have complete samples (6 values each)
        num_samples = len(data) // 6
        data = data[:num_samples * 6]
        
        # Reshape to [samples x 6 channels]
        samples = data.reshape(-1, 6)
        
        # Convert to physical units
        # Accelerometer: raw * 9.807 / 4096 → m/s²
        # Gyroscope: raw * π / (32 * 180) → rad/s
        result = {
            'Ax': samples[:, 0] * 9.807 / 4096,  # Acceleration X-axis (m/s²)
            'Ay': samples[:, 1] * 9.807 / 4096,  # Acceleration Y-axis (m/s²)
            'Az': samples[:, 2] * 9.807 / 4096,  # Acceleration Z-axis (m/s²)
            'Gx': samples[:, 3] * 3.14159 / (32 * 180),  # Gyroscope X-axis (rad/s)
            'Gy': samples[:, 4] * 3.14159 / (32 * 180),  # Gyroscope Y-axis (rad/s)
            'Gz': samples[:, 5] * 3.14159 / (32 * 180),  # Gyroscope Z-axis (rad/s)
        }
        return result
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None


def main() -> None:
    """
    Main function to decode all binary IMU files.
    
    Process:
    1. Read all folders in data/raw/EncryptedData/
    2. Decode each binary file in each folder
    3. Save as CSV files in Decode_Dataset/DecodedData/ with same folder structure
    4. Print summary statistics
    """
    # Determine paths based on current working directory
    current_dir = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    # Check if running from the Sample_Dataset_Acc_Gyro folder
    if current_dir.name == 'Sample_Dataset_Acc_Gyro':
        project_root = current_dir.parent
        input_dir = project_root / 'data' / 'raw' / 'EncryptedData'
        output_dir = current_dir / 'DecodedData'
    else:
        # Running from project root (or elsewhere) — use script location
        project_root = script_dir.parent
        input_dir = project_root / 'data' / 'raw' / 'EncryptedData'
        output_dir = script_dir / 'DecodedData'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: {input_dir} folder not found!")
        print("\nExpected project structure:")
        print("  - From project root: data/raw/EncryptedData/")
        print("  - From Decode_Dataset: ../data/raw/EncryptedData/")
        print(f"\nCurrent directory: {current_dir}")
        return
    
    # Statistics counters
    total_files = 0
    total_samples = 0
    skipped = 0
    
    print("Starting decoding process...")
    print(f"Input: {input_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}\n")
    
    # Process each patient/user folder
    for user_folder in sorted(input_dir.iterdir()):
        if not user_folder.is_dir():
            continue
        
        # Create corresponding output folder
        user_output = output_dir / user_folder.name
        user_output.mkdir(exist_ok=True)
        
        # Process each binary file in user folder
        for data_file in sorted(user_folder.iterdir()):
            if data_file.is_dir():
                continue
            
            # Decode binary file
            data = decode_and_convert(data_file)
            
            if data is None:
                skipped += 1
                continue
            
            # Save as CSV file
            csv_file = user_output / f'{data_file.name}.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header with timestamp
                writer.writerow(['timestamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])

                # Write data rows with timestamp (32 Hz sampling rate)
                sampling_rate = 32
                for i in range(len(data['Ax'])):
                    timestamp = i / sampling_rate  # Convert sample index to seconds
                    writer.writerow([
                        timestamp,
                        data['Ax'][i],
                        data['Ay'][i],
                        data['Az'][i],
                        data['Gx'][i],
                        data['Gy'][i],
                        data['Gz'][i]
                    ])
            
            total_files += 1
            total_samples += len(data['Ax'])
            
            # Progress update every 500 files
            if total_files % 500 == 0:
                print(f'Processed {total_files} files...')
    
    # Print final summary
    print(f'\n{"="*50}')
    print(f'Decoding Complete!')
    print(f'{"="*50}')
    print(f'✓ Successfully decoded: {total_files} files')
    print(f'✗ Skipped (invalid): {skipped} files')
    print(f'📊 Total samples: {total_samples:,}')
    print(f'📁 Output directory: {output_dir.absolute()}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()