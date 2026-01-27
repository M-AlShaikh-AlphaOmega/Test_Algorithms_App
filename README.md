# Parkinson's Detection Pipeline - Random Forest Specialized

## 1. Project Overview
This project specializes in detecting the **"OFF" state** in Parkinson's patients using wearable IMU (Inertial Measurement Unit) sensor data. By analyzing accelerometer and gyroscope signals, the pipeline identifies motor states (ON vs OFF medication) with high precision using machine learning.

## 2. Project Structure
```
Test_Algorithms_App/
├── RandomForest.py              # ML pipeline for decoded_csv data (recommended)
├── RandomForest_Test_Train.py   # ML pipeline for Data/ folder (legacy)
├── Decode_Dataset.py            # Decodes raw binary IMU data to CSV
├── labels_template.csv          # Template for user labels
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── Data/                        # Processed CSV data for ML training
├── rw_backup/                   # Raw binary IMU data (place here before decoding)
├── decoded_csv/                 # Output folder for decoded CSV files
├── outputs/                     # ML pipeline results and visualizations
└── venv/                        # Python virtual environment
```

## 3. Environment Setup
To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Test_Algorithms_App
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows**: `.\venv\Scripts\activate`
   - **macOS/Linux**: `source venv/bin/activate`

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 4. Technology Stack & Libraries
The project is built using the Python scientific stack:
- **NumPy & Pandas**: Data manipulation and numerical operations.
- **SciPy**: Signal processing (Butterworth filters) and statistical analysis (Skewness, Kurtosis).
- **Scikit-Learn**: Machine learning pipeline, Random Forest implementation, and evaluation metrics.
- **Matplotlib**: Generation of performance visualizations (Confusion Matrix, ROC Curve).

## 5. Software Architecture & Design Patterns
The codebase is designed for modularity and extensibility using modern software engineering patterns:

### Design Patterns
- **Strategy Pattern**: used in `BaseExtractor`, `StatExtractor`, and `GaitExtractor`. This allows the pipeline to swap or add new feature extraction logic without modifying the core `Pipeline` execution logic.
- **Composition**: The `Pipeline` class coordinates specialized objects (`Preprocessor`, `WindowSegmenter`, `BaseExtractor` list) rather than using complex inheritance.
- **Config Pattern**: Uses Python `dataclasses` (`PipelineConfig`, `FilterConfig`) to centralize parameters, making the code clean and easy to tune.

### SOLID Principles
- **Single Responsibility Principle (SRP)**: Each class has one job. `ButterworthFilter` only filters, `Preprocessor` manages signal flow, and `WindowSegmenter` only slices data.
- **Open/Closed Principle (OCP)**: New metrics or sensors can be added by creating new subclasses of `BaseExtractor` without changing existing code.
- **Liskov Substitution Principle (LSP)**: All extractor subclasses implement the `extract` method, ensuring they can be used interchangeably by the pipeline.
- **Dependency Inversion**: The `Pipeline` depends on abstractions (like the list of extractors) rather than concrete implementations.

## 6. Technical Flow
1. **Signal Conditioning**: Raw IMU data is processed via a 4th-order Butterworth low-pass filter (15Hz cutoff) to remove high-frequency noise and isolate human motion.
2. **Windowing**: Data is segmented into fixed-size windows (e.g., 320 samples) for localized analysis.
3. **Feature Extraction**: 
   - **Time-Domain**: Mean, Std, RMS, IQR, etc. (36 features total).
   - **Gait-Specific**: Step detection via peak finding on the Y-axis, cadence calculation, and arm swing magnitude.
4. **Classification**: A **Random Forest Classifier** (100 estimators) is trained on the extracted features. Random Forest was chosen for its ability to handle non-linear relationships in physical movement data and its resistance to overfitting.

## 7. Usage Guide
Run the pipeline with automated labeling:
```bash
python RandomForest_Test_Train.py --fast
```
### CLI Arguments:
- `--fs`: Set sampling frequency (default 32Hz).
- `--data`: Path to the Data folder.
- `--task`: Task name to process (e.g., `walk_hold_left`).
- `--fast`: Fast mode (disables plots).

## 8. Data & Labeling
- **Structure**: Expects a `Data/` folder with subdirectories per case. Each case must have matching `acc` and `gyro` CSV files.
- **Auto-Labeling**: Labels are automatically inferred from folder suffixes (e.g., `mnsy`).
    - 2nd character **'n'**: Medication OFF (Class 1).
    - 2nd character **'y'**: Medication ON (Class 0).

## 9. Raw Data Decoding (rw_backup)
The `rw_backup/` folder contains raw binary IMU data files collected from wearable devices.

### Prerequisites
> **IMPORTANT:** You must place the `rw_backup/` folder in the **same directory** as `Decode_Dataset.py` before running the script.

**Required folder structure:**
```
Test_Algorithms_App/
├── Decode_Dataset.py
├── RandomForest_Test_Train.py
├── rw_backup/                  <-- Place here
│   ├── <user_id_1>/
│   │   ├── 2025_06_09_14_08_17
│   │   ├── 2025_06_09_14_08_27
│   │   └── ...
│   └── <user_id_2>/
│       └── ...
├── Data/
├── outputs/
└── ...
```

### Decode_Dataset.py
This script processes binary IMU data files from the `rw_backup/` directory, decodes them into physical units, and saves the results as CSV files.

**Binary Format**:
- Data stored as **float32** (4 bytes per value)
- **6 values per sample**: Ax, Ay, Az, Gx, Gy, Gz
- Accelerometer conversion: `raw * 9.807 / 4096` (m/s²)
- Gyroscope conversion: `raw * π / (32 * 180)` (rad/s)
- Sampling rate: **32 Hz**

**Run the decoder:**
```bash
python Decode_Dataset.py
```

**Output Structure:**
```
decoded_csv/
├── <user_id_1>/
│   ├── 2025_06_09_14_08_17.csv
│   ├── 2025_06_09_14_08_27.csv
│   └── ...
├── <user_id_2>/
│   └── ...
└── ...
```

**CSV Format:**
| Sample | Ax     | Ay     | Az     | Gx     | Gy      | Gz     |
|--------|--------|--------|--------|--------|---------|--------|
| 0      | 0.6704 | 1.5395 | 9.6993 | 0.0180 | -0.0660 | 0.0016 |
| 1      | ...    | ...    | ...    | ...    | ...     | ...    |

- **Ax, Ay, Az**: Accelerometer values in m/s²
- **Gx, Gy, Gz**: Gyroscope values in rad/s

### Usage Examples

**1. Decode all rw_backup files to CSV:**
```bash
python Decode_Dataset.py
```

**2. Load decoded CSV for analysis:**
```python
import pandas as pd

# Load a decoded file
df = pd.read_csv("decoded_csv/oA9zb5Jvsia7Tl9fTnqhYzLhkXUo/2025_06_09_14_08_17.csv")

# Access accelerometer data
ax = df['Ax'].values  # m/s²
ay = df['Ay'].values
az = df['Az'].values

# Access gyroscope data
gx = df['Gx'].values  # rad/s
gy = df['Gy'].values
gz = df['Gz'].values
```

**3. Use decoded data with the ML pipeline:**
```python
import pandas as pd

# Load decoded data
df = pd.read_csv("decoded_csv/user_id/timestamp.csv")

# Prepare for pipeline (rename columns to match expected format)
acc_df = pd.DataFrame({'X': df['Ax'], 'Y': df['Ay'], 'Z': df['Az']})
gyro_df = pd.DataFrame({'X': df['Gx'], 'Y': df['Gy'], 'Z': df['Gz']})

# Now you can use these with the Preprocessor and feature extractors
```

**4. Use the decode function in your own code:**
```python
from Decode_Dataset import decode_and_convert

file_path = "rw_backup/oA9zb5Jvsia7Tl9fTnqhYzLhkXUo/2025_06_09_14_08_17"
data = decode_and_convert(file_path)
if data is not None:
    print(f"Decoded {len(data['Ax'])} samples")
    print(f"Ax range: {data['Ax'].min():.3f} to {data['Ax'].max():.3f} m/s²")
```

## 10. Training with Decoded Data (RandomForest.py)
This is the recommended pipeline for training on decoded IMU data from `decoded_csv/`.

### Prerequisites
1. Decode raw data first: `python Decode_Dataset.py`
2. Create a labels file mapping user IDs to classes

### Labels File Format
Create a CSV file (e.g., `labels.csv`) with user labels:
```csv
user_id,label
oA9zb5Jvsia7Tl9fTnqhYzLhkXUo,1
akwn2joina1,0
oA9zb5Au-No3klQwjuY28POhL7U0,1
```
- `user_id`: Folder name in `decoded_csv/`
- `label`: 0 = ON medication, 1 = OFF medication

A template is provided: `labels_template.csv`

### Run the Pipeline
```bash
python RandomForest.py --labels labels.csv
```

### CLI Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `decoded_csv` | Path to decoded CSV directory |
| `--output` | `outputs` | Path to output directory |
| `--labels` | (required) | Path to labels CSV file |
| `--fs` | `32` | Sampling frequency in Hz |
| `--window` | `320` | Window size in samples |
| `--features` | `15` | Number of top features to select |
| `--fast` | `false` | Skip plot generation |

### Examples
```bash
# Basic usage
python RandomForest.py --labels labels.csv

# Custom settings
python RandomForest.py --data decoded_csv --labels labels.csv --window 256 --features 20

# Fast mode (no plots)
python RandomForest.py --labels labels.csv --fast
```

### Architecture & Design Patterns
The `RandomForest.py` pipeline implements:

**Design Patterns:**
- **Strategy Pattern**: `FeatureExtractor` abstract class with `StatisticalExtractor`, `GaitExtractor`, `FrequencyExtractor`
- **Composition**: `Pipeline` coordinates `Preprocessor`, `WindowSegmenter`, `FeatureExtractorCoordinator`, `ModelManager`
- **Config Pattern**: Dataclasses (`PipelineConfig`, `FilterConfig`, `WindowConfig`, `ModelConfig`)

**SOLID Principles:**
- **SRP**: Each class has one responsibility (e.g., `DataLoader` only loads data)
- **OCP**: Add new extractors without modifying existing code
- **LSP**: All extractors implement `FeatureExtractor` interface
- **DIP**: `Pipeline` depends on abstractions, not concrete implementations

**Features Extracted:**
| Category | Features |
|----------|----------|
| Statistical | mean, std, min, max, range, rms, skew, kurtosis, iqr, median |
| Gait | steps, step_time, cadence, step_regularity, arm_swing |
| Frequency | dominant_freq, spectral_energy, spectral_centroid |

## 11. Outputs & File Explanations
Results are saved in the `outputs/` folder. Each file serves a specific role in the analysis:

- **`extracted_features.csv`**: 
  - *What it is*: A massive table containing all calculated metrics (Mean, RMS, Cadence, etc.) for every data window processed.
  - *Why we need it*: It serves as the "Gold Standard" dataset. It allows you to inspect the raw numbers before they enter the AI model and can be reused for training other models (like SVM or Neural Networks) without re-running the extraction.

- **`evaluation_report.txt`**: 
  - *What it is*: A text summary showing the **Precision, Recall, F1-Score**, and **Cross-Validation Accuracy**.
  - *Why we need it*: This is your primary "Health Check." It tells you exactly how reliable the AI is. The *Cross-Validation* score specifically ensures that the model's performance is consistent and not just a result of a lucky guess.

- **`feature_importance.png`**: 
  - *What it is*: A bar chart ranking which sensor metrics (e.g., `Y_rms` or `cadence`) had the most influence on the "OFF" state detection.
  - *Why we need it*: It provides **Explainability**. In medical apps, you need to know *why* a model made a decision. If `arm_swing` is high on the list, it proves the model is looking at the correct physical symptoms.

- **`confusion_matrix.png`**: 
  - *What it is*: A grid showing exactly how many "ON" states were correctly identified and how many were mistaken for "OFF" (and vice-versa).
  - *Why we need it*: Accuracy doesn't tell the whole story. This plot helps you see if the model is biased (e.g., if it's very good at finding "OFF" but constantly misses "ON" states).

- **`roc_curve.png`**: 
  - *What it is*: A "Receiver Operating Characteristic" curve that plots the True Positive Rate against the False Positive Rate.
  - *Why we need it*: It measures the model's ability to distinguish between classes at various thresholds. A curve that bows toward the top-left corner indicates a high-performing, stable classifier.


