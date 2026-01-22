# Parkinson's Detection Pipeline - Random Forest Specialized

## 1. Project Overview
This project specializes in detecting the **"OFF" state** in Parkinson's patients using wearable IMU (Inertial Measurement Unit) sensor data. By analyzing accelerometer and gyroscope signals, the pipeline identifies motor states (ON vs OFF medication) with high precision using machine learning.

## 2. Environment Setup
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

## 3. Technology Stack & Libraries
The project is built using the Python scientific stack:
- **NumPy & Pandas**: Data manipulation and numerical operations.
- **SciPy**: Signal processing (Butterworth filters) and statistical analysis (Skewness, Kurtosis).
- **Scikit-Learn**: Machine learning pipeline, Random Forest implementation, and evaluation metrics.
- **Matplotlib**: Generation of performance visualizations (Confusion Matrix, ROC Curve).

## 4. Software Architecture & Design Patterns
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

## 5. Technical Flow
1. **Signal Conditioning**: Raw IMU data is processed via a 4th-order Butterworth low-pass filter (15Hz cutoff) to remove high-frequency noise and isolate human motion.
2. **Windowing**: Data is segmented into fixed-size windows (e.g., 320 samples) for localized analysis.
3. **Feature Extraction**: 
   - **Time-Domain**: Mean, Std, RMS, IQR, etc. (36 features total).
   - **Gait-Specific**: Step detection via peak finding on the Y-axis, cadence calculation, and arm swing magnitude.
4. **Classification**: A **Random Forest Classifier** (100 estimators) is trained on the extracted features. Random Forest was chosen for its ability to handle non-linear relationships in physical movement data and its resistance to overfitting.

## 6. Usage Guide
Run the pipeline with automated labeling:
```bash
python RandomForest_Test_Train.py --fast
```
### CLI Arguments:
- `--fs`: Set sampling frequency (default 32Hz).
- `--data`: Path to the Data folder.
- `--task`: Task name to process (e.g., `walk_hold_left`).
- `--fast`: Fast mode (disables plots).

## 7. Data & Labeling
- **Structure**: Expects a `Data/` folder with subdirectories per case. Each case must have matching `acc` and `gyro` CSV files.
- **Auto-Labeling**: Labels are automatically inferred from folder suffixes (e.g., `mnsy`).
    - 2nd character **'n'**: Medication OFF (Class 1).
    - 2nd character **'y'**: Medication ON (Class 0).

## 8. Outputs & File Explanations
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


