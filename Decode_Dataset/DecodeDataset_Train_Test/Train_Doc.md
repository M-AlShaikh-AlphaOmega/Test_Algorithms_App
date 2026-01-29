# Random Forest Training Documentation
## aCare Project - OFF State Detection Model Training

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Data Organization](#data-organization)
3. [How to Label Data (ON vs OFF)](#how-to-label-data)
4. [Parameter Explanations](#parameter-explanations)
5. [Feature Extraction](#feature-extraction)
6. [Training Process](#training-process)
7. [How to Run](#how-to-run)
8. [Understanding Results](#understanding-results)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This training pipeline builds a **Random Forest classifier** that detects OFF states in Parkinson's disease patients using IMU sensor data from the decoded dataset.

**Goal**: Classify patient movement data into two categories:
- **ON State (Label 0)**: Patient's medications are working well, good mobility
- **OFF State (Label 1)**: Medications wearing off, movement difficulties

**Expected Accuracy**: 93-96% (based on research literature)

---

## 📁 Data Organization

### Required Folder Structure

You must organize your decoded CSV files into two folders:

```
data/
└── Decode_Dataset/
    ├── DecodedData/                    # Original decoded files (don't modify)
    │   ├── patient_1/
    │   ├── patient_2/
    │   └── ...
    └── DecodeDataset_Train_Test/       # Create this for training
        ├── ON/                          # Put ON state CSV files here
        │   ├── patient1_morning.csv
        │   ├── patient2_afternoon.csv
        │   └── ...
        ├── OFF/                         # Put OFF state CSV files here
        │   ├── patient1_evening.csv
        │   ├── patient3_morning.csv
        │   └── ...
        └── outputs/                     # Results will be saved here (auto-created)
            ├── extracted_features.csv
            ├── evaluation_report.txt
            ├── feature_importance.png
            ├── confusion_matrix.png
            └── roc_curve.png
```

### Creating Training Folders

**Step 1**: Create the folder structure
```bash
cd data/Decode_Dataset
mkdir -p DecodeDataset_Train_Test/ON
mkdir -p DecodeDataset_Train_Test/OFF
```

**Step 2**: Copy labeled CSV files
- Copy ON state recordings → `DecodeDataset_Train_Test/ON/`
- Copy OFF state recordings → `DecodeDataset_Train_Test/OFF/`

---

## 🏷️ How to Label Data (ON vs OFF)

### What is ON State?
**Clinical Definition**: Patient is in "ON state" when:
- ✅ Medications are working effectively
- ✅ Good motor control and coordination
- ✅ Smooth, regular movements
- ✅ Normal or near-normal gait patterns
- ✅ Minimal tremor or bradykinesia

**When to Record ON State**:
- 30-90 minutes after taking levodopa medication
- Patient reports feeling "good" or "normal"
- Clinical assessment shows MDS-UPDRS motor score < 20
- Patient can perform tasks smoothly

**Movement Characteristics**:
- Regular stride length and timing
- Strong arm swing
- Good balance and posture
- Faster walking speed
- Less freezing or hesitation

### What is OFF State?
**Clinical Definition**: Patient is in "OFF state" when:
- ❌ Medications have worn off
- ❌ Reduced motor control
- ❌ Slower, irregular movements
- ❌ Movement drops below 60% of personal baseline
- ❌ Increased tremor or stiffness

**When to Record OFF State**:
- Before medication dose (morning or between doses)
- 3-5 hours after last medication
- Patient reports feeling "off" or movement difficulty
- Clinical assessment shows MDS-UPDRS motor score > 30

**Movement Characteristics**:
- Shorter, irregular stride length
- Reduced or absent arm swing
- Shuffling gait
- Slower walking speed
- Increased freezing episodes
- More tremor and rigidity

### Data Labeling Guidelines

#### Method 1: Clinical Assessment (Most Reliable)
1. **Use MDS-UPDRS Scores**:
   - MDS-UPDRS Motor Score < 25 → ON state
   - MDS-UPDRS Motor Score > 30 → OFF state
   - Scores 25-30 → Transitional (exclude from training)

2. **Use Patient Diary**:
   - Patient records when they took medication
   - Patient notes when they feel "ON" or "OFF"
   - Clinician validates patient reports

3. **Use Medication Timing**:
   - 30-90 min after medication → likely ON
   - 3-5 hours after medication → likely OFF
   - Just before next dose → likely OFF

#### Method 2: Movement Analysis (If No Clinical Data)
Compare patient's movement to their personal baseline:
- Movement intensity ≥ 80% of baseline → ON state
- Movement intensity ≤ 60% of baseline → OFF state
- Between 60-80% → Uncertain (exclude)

**Calculate Baseline**:
```python
# Calculate patient's average movement when feeling best
baseline_movement = mean(movement_intensity_during_best_times)

# For each recording:
if current_movement >= 0.8 * baseline_movement:
    label = "ON"
elif current_movement <= 0.6 * baseline_movement:
    label = "OFF"
else:
    label = "UNCERTAIN"  # Don't use for training
```

### Important Notes on Labeling

**⚠️ Quality over Quantity**:
- Better to have 100 correctly labeled samples than 500 uncertain ones
- When in doubt, exclude the recording from training
- Mislabeled data hurts model accuracy more than having less data

**⚠️ Balance Your Dataset**:
- Try to have similar amounts of ON and OFF data
- Ideal: 50% ON state, 50% OFF state
- Minimum: 40% of either class
- Too imbalanced (e.g., 90% ON, 10% OFF) will bias the model

**⚠️ Patient Diversity**:
- Include data from multiple patients
- Different disease severity levels
- Different medication responses
- Various times of day

---

## ⚙️ Parameter Explanations

### Why These Values?

#### 1. Sampling Frequency: `fs = 32 Hz`
**What it means**: 32 measurements per second

**Why 32 Hz?**:
- ✅ Captures human movement frequencies (0.5-10 Hz)
- ✅ Gait cycles are ~1-2 Hz (0.5-1 sec per step)
- ✅ Tremor is 4-6 Hz (Parkinson's typical)
- ✅ Matches your Oppo Watch sensor rate
- ❌ Higher rates (100+ Hz) capture noise, not useful signals
- ❌ Lower rates (<20 Hz) miss important movement details

**Reference**: Research shows 25-50 Hz is optimal for gait analysis

#### 2. Window Size: `window_size_sec = 10 seconds`
**What it means**: Analyze data in 10-second chunks

**Why 10 seconds?**:
- ✅ Captures 8-12 complete steps (at ~1 step/sec)
- ✅ Long enough to detect gait patterns
- ✅ Short enough for real-time detection
- ✅ Standard in Parkinson's research literature
- ❌ Too short (<5 sec): Misses gait cycles
- ❌ Too long (>30 sec): Delays detection, mixes states

**Calculation**:
```
Window samples = fs × window_size_sec = 32 Hz × 10 sec = 320 samples
```

#### 3. Filter Cutoff: `cutoff = 15 Hz`
**What it means**: Remove frequencies above 15 Hz

**Why 15 Hz?**:
- ✅ Human movement: 0.5-10 Hz (keep)
- ✅ Tremor: 4-6 Hz (keep)
- ✅ High-frequency noise: >15 Hz (remove)
- ✅ Prevents aliasing artifacts
- ❌ Too low (<10 Hz): Removes useful tremor signals
- ❌ Too high (>20 Hz): Keeps sensor noise

**Nyquist frequency**: 32 Hz ÷ 2 = 16 Hz (maximum detectable)

#### 4. Filter Order: `order = 4`
**What it means**: Complexity of the filter

**Why 4th order?**:
- ✅ Good balance: effective filtering without distortion
- ✅ Butterworth filter provides flat passband
- ✅ Standard in biomedical signal processing
- ❌ Lower order (<3): Weak filtering
- ❌ Higher order (>6): Can cause artifacts

#### 5. Number of Trees: `n_estimators = 100`
**What it means**: Random Forest uses 100 decision trees

**Why 100 trees?**:
- ✅ Stable, reliable predictions
- ✅ Diminishing returns beyond 100
- ✅ Good speed/accuracy tradeoff
- ✅ Prevents overfitting better than single tree
- ❌ Too few (<50): Unstable predictions
- ❌ Too many (>200): Slower, no accuracy gain

**Each tree votes** → Final prediction = majority vote

#### 6. Max Depth: `max_depth = 7`
**What it means**: Decision trees can be 7 levels deep

**Why depth 7?**:
- ✅ Can learn 2^7 = 128 decision rules
- ✅ Prevents overfitting (memorizing training data)
- ✅ Ensures model generalizes to new patients
- ❌ Too shallow (<5): Underfitting, poor accuracy
- ❌ Too deep (>10): Overfitting, memorizes noise

**Overfitting check**: If training accuracy >> test accuracy, reduce depth

#### 7. Min Samples per Leaf: `min_samples_leaf = 2`
**What it means**: Each leaf node must have ≥2 samples

**Why 2 samples?**:
- ✅ Prevents creating rules for single outliers
- ✅ Ensures statistical reliability
- ✅ Balances precision with generalization
- ❌ Too low (=1): Overfitting to noise
- ❌ Too high (>5): Underfitting, too simple

#### 8. Max Features: `max_features = 'sqrt'`
**What it means**: Each tree sees √(total features) random features

**Why sqrt?**:
- ✅ If 15 features total → each tree sees √15 ≈ 4 features
- ✅ Creates diverse trees (different feature subsets)
- ✅ Prevents correlation between trees
- ✅ Improves ensemble performance
- ❌ 'log2': Too restrictive
- ❌ 'None': All features → correlated trees

#### 9. Test Size: `test_size = 0.3`
**What it means**: 30% of data for testing, 70% for training

**Why 70/30 split?**:
- ✅ 70% provides enough data to learn patterns
- ✅ 30% provides reliable performance estimate
- ✅ Standard machine learning practice
- ✅ Ensures model isn't tested on training data
- ❌ 90/10: Unreliable test results (too small)
- ❌ 50/50: Not enough training data

#### 10. Cross-Validation Folds: `n_splits = 5`
**What it means**: Data split 5 ways for validation

**Why 5-fold CV?**:
- ✅ Each data point tested exactly once
- ✅ Training uses 80% each fold
- ✅ Reliable performance estimate
- ✅ Detects overfitting
- ❌ 3-fold: Less reliable estimate
- ❌ 10-fold: Slower, similar results

**How it works**:
```
Fold 1: Train[2,3,4,5] Test[1]
Fold 2: Train[1,3,4,5] Test[2]
Fold 3: Train[1,2,4,5] Test[3]
Fold 4: Train[1,2,3,5] Test[4]
Fold 5: Train[1,2,3,4] Test[5]

Final score = Average of 5 test scores
```

#### 11. Top Features: `top_k_features = 15`
**What it means**: Select 15 most important features

**Why 15 features?**:
- ✅ Reduces noise from irrelevant features
- ✅ Faster training and inference
- ✅ Easier to interpret results
- ✅ Prevents curse of dimensionality
- ✅ Research shows 10-20 features optimal for gait analysis
- ❌ Too few (<10): Loses important information
- ❌ Too many (>30): Includes noise, slower

**Feature selection prevents overfitting** by removing weak predictors

#### 12. Random Seed: `seed = 42`
**What it means**: Fixed random number generator

**Why use a seed?**:
- ✅ Reproducible results
- ✅ Same train/test split every run
- ✅ Compare different models fairly
- ✅ Debug issues consistently
- ❌ Without seed: Different results each run

**Why 42?**: Arbitrary convention (from "Hitchhiker's Guide to the Galaxy")

---

## 🔬 Feature Extraction

### What Are Features?

**Features** are numerical measurements that describe movement patterns. The model learns which features best distinguish ON from OFF states.

### Statistical Features (27 features)

#### Accelerometer Features (9 × 3 axes = 27 features)

**For each axis (Ax, Ay, Az)**:

1. **Mean** (`acc_Ax_mean`):
   - Average acceleration
   - **Purpose**: Overall movement intensity
   - **ON state**: Higher values (more movement)
   - **OFF state**: Lower values (less movement)

2. **Standard Deviation** (`acc_Ax_std`):
   - Movement variability
   - **Purpose**: Consistency of movement
   - **ON state**: Moderate, regular patterns
   - **OFF state**: Higher (irregular) or lower (rigid)

3. **Min/Max** (`acc_Ax_min`, `acc_Ax_max`):
   - Extreme values
   - **Purpose**: Range of motion
   - **ON state**: Wider range
   - **OFF state**: Narrower range (restricted movement)

4. **Range** (`acc_Ax_range`):
   - Max - Min
   - **Purpose**: Movement amplitude
   - **ON state**: Larger range
   - **OFF state**: Smaller range

5. **RMS** (`acc_Ax_rms`):
   - Root Mean Square = √(mean(signal²))
   - **Purpose**: Overall energy of movement
   - **ON state**: Higher energy
   - **OFF state**: Lower energy

6. **Skewness** (`acc_Ax_skew`):
   - Distribution asymmetry
   - **Purpose**: Detect irregular movement patterns
   - **Normal**: Near 0 (symmetric)
   - **Abnormal**: Large positive/negative (asymmetric gait)

7. **Kurtosis** (`acc_Ax_kurt`):
   - Distribution "peakedness"
   - **Purpose**: Detect erratic movements
   - **Normal**: Near 3 (gaussian)
   - **Abnormal**: High (spiky, tremor) or low (flat, rigid)

8. **IQR** (`acc_Ax_iqr`):
   - Interquartile Range (75th - 25th percentile)
   - **Purpose**: Robust measure of variability
   - **Advantage**: Less affected by outliers than std

#### Gyroscope Features (27 features)
Same 9 features for each axis (Gx, Gy, Gz):
- Measures **rotation** instead of linear movement
- Important for detecting **arm swing**, **turning**, **balance**

**Why gyroscope matters**:
- OFF state shows **reduced arm swing** (less rotation)
- OFF state shows **difficulty turning** (abnormal Gz)
- OFF state shows **postural instability** (irregular Gx, Gy)

### Gait Features (4 features)

#### 1. Steps (`steps`)
**What it measures**: Number of steps detected in 10-second window

**How detected**: Peak finding on vertical acceleration (Ay)
- Peaks = heel strikes
- `prominence=0.5`: Minimum 0.5 m/s² for valid step
- `distance=16`: Minimum 0.5 sec between steps

**Typical values**:
- **ON state**: 8-12 steps in 10 seconds
- **OFF state**: 4-8 steps (slower walking)

#### 2. Step Time (`step_time`)
**What it measures**: Average time between steps (seconds)

**Calculation**: 
```python
step_time = mean(time_between_peaks)
```

**Typical values**:
- **ON state**: 0.8-1.2 seconds (normal cadence)
- **OFF state**: 1.2-2.0 seconds (slower steps)

**Clinical relevance**: Bradykinesia (slowness) increases step time

#### 3. Cadence (`cadence`)
**What it measures**: Steps per minute

**Calculation**:
```python
cadence = (number_of_steps / window_duration) × 60
```

**Typical values**:
- **ON state**: 90-120 steps/min (normal walking)
- **OFF state**: 50-80 steps/min (shuffling gait)
- **Healthy adults**: 100-120 steps/min

**Clinical relevance**: Key indicator of Parkinson's severity

#### 4. Arm Swing (`arm_swing`)
**What it measures**: Range of arm rotation during walking

**Calculation**:
```python
gyro_magnitude = √(Gx² + Gy² + Gz²)
arm_swing = max(gyro_magnitude) - min(gyro_magnitude)
```

**Typical values**:
- **ON state**: 2.0-5.0 rad/s (normal arm movement)
- **OFF state**: 0.5-2.0 rad/s (reduced/absent swing)
- **Severe OFF**: <0.5 rad/s (frozen arms)

**Clinical relevance**: 
- One of earliest Parkinson's signs
- Often asymmetric (one arm affected more)
- Strong predictor of OFF state

### Total Features Extracted

Before feature selection:
- **Accelerometer stats**: 27 features
- **Gyroscope stats**: 27 features
- **Gait metrics**: 4 features
- **Total**: 58 features

After feature selection:
- **Top 15 most important features** are selected
- These are usually: Ay_std, arm_swing, cadence, steps, Az_mean, Gx_std, etc.

---

## 🎓 Training Process

### Step-by-Step Pipeline

#### 1. Data Loading
```
Load CSV files from ON/ and OFF/ folders
↓
Validate: Check for required columns (Ax, Ay, Az, Gx, Gy, Gz)
↓
Drop rows with missing values
↓
Check minimum length (≥320 samples for 10-sec window)
```

#### 2. Preprocessing
```
Apply Butterworth low-pass filter (15 Hz cutoff)
↓
Remove high-frequency noise
↓
Preserve movement signals (0.5-10 Hz)
```

**Why filter?**:
- Sensor noise can confuse the model
- Filter keeps useful signals, removes artifacts
- Improves feature quality

#### 3. Windowing
```
Segment each file into 10-second windows
↓
Window size = 32 Hz × 10 sec = 320 samples
↓
Non-overlapping windows (no data reuse)
```

**Example**:
- File with 1600 samples (50 seconds) → 5 windows
- Each window is independent training example

#### 4. Feature Extraction
```
For each window:
├─ Extract statistical features from accelerometer
├─ Extract statistical features from gyroscope
└─ Extract gait features (steps, cadence, arm swing)
↓
Combine into feature vector (58 features per window)
```

#### 5. Feature Selection
```
Train preliminary Random Forest on all 58 features
↓
Calculate feature importance scores
↓
Select top 15 features
↓
These features are used for final model
```

**Why select features?**:
- Reduces overfitting
- Faster training and inference
- Easier to interpret
- Improves generalization

#### 6. Train-Test Split
```
Split data: 70% training, 30% testing
↓
Stratified split (maintain ON/OFF ratio)
↓
Training set: Model learns patterns
Testing set: Evaluate on unseen data
```

**Stratified split example**:
- Total: 1000 ON, 1000 OFF
- Training: 700 ON, 700 OFF
- Testing: 300 ON, 300 OFF

#### 7. Model Training
```
Train Random Forest with 100 trees
↓
Each tree learns different patterns
↓
Trees vote on final prediction
```

#### 8. Evaluation
```
Test on held-out test set
↓
5-fold cross-validation
↓
Calculate metrics: Accuracy, Precision, Recall, AUC
```

### Cross-Validation Explained

**Purpose**: Ensure model works on different data subsets

**Process**:
```
Original Data: [1][2][3][4][5]

Fold 1: Test[1]  Train[2,3,4,5] → Accuracy₁
Fold 2: Test[2]  Train[1,3,4,5] → Accuracy₂  
Fold 3: Test[3]  Train[1,2,4,5] → Accuracy₃
Fold 4: Test[4]  Train[1,2,3,5] → Accuracy₄
Fold 5: Test[5]  Train[1,2,3,4] → Accuracy₅

Final CV Score = Mean(Accuracy₁...₅)
```

**Good CV score**: All folds have similar accuracy
**Bad CV score**: Large variance between folds (overfitting)

---

## 🚀 How to Run

### Prerequisites

**1. Install Required Packages**:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

**2. Prepare Your Data**:
- Decode your raw sensor files (use `Decode_Dataset.py`)
- Label and organize into ON/ and OFF/ folders
- Ensure balanced dataset (similar amounts of ON and OFF)

### Running the Training Script

#### Option 1: Use Default Paths
```bash
cd Decode_Dataset/DecodeDataset_Train_Test
python ../../RandomForest_Train.py
```

#### Option 2: Specify Custom Paths
```bash
python RandomForest_Train.py \
    --data_on path/to/ON_data \
    --data_off path/to/OFF_data \
    --out path/to/outputs
```

#### Option 3: Customize Parameters
```bash
python RandomForest_Train.py \
    --data_on Decode_Dataset/DecodeDataset_Train_Test/ON \
    --data_off Decode_Dataset/DecodeDataset_Train_Test/OFF \
    --out Decode_Dataset/DecodeDataset_Train_Test/outputs \
    --window_sec 10 \
    --top_k 15 \
    --fs 32 \
    --fast
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_on` | `data/Decode_Dataset/DecodeDataset_Train_Test/ON` | Path to ON state CSV files |
| `--data_off` | `data/Decode_Dataset/DecodeDataset_Train_Test/OFF` | Path to OFF state CSV files |
| `--out` | `data/Decode_Dataset/DecodeDataset_Train_Test/outputs` | Output directory |
| `--fs` | `32` | Sampling frequency (Hz) |
| `--window_sec` | `10` | Window size (seconds) |
| `--top_k` | `15` | Number of features to select |
| `--fast` | `False` | Skip plots (faster) |

### Expected Output

```
============================================================
Starting Random Forest Training Pipeline
============================================================
2026-01-29 10:30:00 - INFO - Starting data extraction...
2026-01-29 10:30:05 - INFO - Processing 245 files from ON...
2026-01-29 10:30:25 - INFO - Extracted 1234 windows from ON state data
2026-01-29 10:30:26 - INFO - Processing 238 files from OFF...
2026-01-29 10:30:45 - INFO - Extracted 1198 windows from OFF state data
2026-01-29 10:30:45 - INFO - Total windows extracted: 2432
2026-01-29 10:30:45 - INFO - ON state windows: 1234
2026-01-29 10:30:45 - INFO - OFF state windows: 1198
2026-01-29 10:30:46 - INFO - Saved extracted features to outputs/extracted_features.csv
2026-01-29 10:30:46 - INFO - Total features: 58
2026-01-29 10:30:46 - INFO - Selecting top 15 features from 58 total features...
2026-01-29 10:30:48 - INFO - Selected features: ['acc_Ay_std', 'arm_swing', 'cadence', ...]
2026-01-29 10:30:48 - INFO - Training set: 1702 samples
2026-01-29 10:30:48 - INFO - Test set: 730 samples
2026-01-29 10:30:50 - INFO - Training Accuracy: 0.9847
2026-01-29 10:30:50 - INFO - Test Accuracy:     0.9521
2026-01-29 10:30:52 - INFO - CV Mean Accuracy:  0.9487 (+/- 0.0124)
2026-01-29 10:30:52 - INFO - AUC Score: 0.9823
2026-01-29 10:30:52 - INFO - Saved evaluation report to outputs/evaluation_report.txt
2026-01-29 10:30:55 - INFO - Generating visualization plots...
2026-01-29 10:30:58 - INFO - Saved plots to outputs
============================================================
Pipeline complete! Results saved in outputs
============================================================
```

---

## 📊 Understanding Results

### Output Files

#### 1. extracted_features.csv
**What it contains**: All extracted features for every window

**Columns**:
- `file`: Source CSV filename
- `window`: Window index
- `label`: 0=ON, 1=OFF
- `acc_Ax_mean`, `acc_Ax_std`, ...: Accelerometer features
- `gyro_Gx_mean`, `gyro_Gx_std`, ...: Gyroscope features
- `steps`, `step_time`, `cadence`, `arm_swing`: Gait features

**Use**: Inspect features, validate extraction, debug issues

#### 2. evaluation_report.txt
**What it contains**: Model performance metrics

**Example**:
```
============================================================
Random Forest - Parkinson's OFF State Detection
============================================================

Training Accuracy: 0.9847
Test Accuracy:     0.9521
CV Mean Accuracy:  0.9487 (+/- 0.0124)
AUC Score:         0.9823

Classification Report:
------------------------------------------------------------
              precision    recall  f1-score   support

          ON       0.96      0.94      0.95       367
         OFF       0.95      0.96      0.95       363

    accuracy                           0.95       730
   macro avg       0.95      0.95      0.95       730
weighted avg       0.95      0.95      0.95       730
```

**Interpreting Metrics**:

- **Accuracy (95.21%)**: Overall correct predictions
  - Target: >93% (research benchmark)
  - If <90%: Check data quality, labels, balance

- **Precision (ON=96%, OFF=95%)**: When predicting ON/OFF, how often correct?
  - High precision = Few false alarms
  - Important for patient confidence

- **Recall (ON=94%, OFF=96%)**: Of actual ON/OFF states, how many detected?
  - High recall = Few missed episodes
  - Important for safety (catch all OFF states)

- **F1-Score (95%)**: Harmonic mean of precision and recall
  - Balanced metric
  - Use when precision and recall both matter

- **AUC (98.23%)**: Area Under ROC Curve
  - Measures discrimination ability
  - 0.5 = Random, 1.0 = Perfect
  - >0.95 = Excellent performance

#### 3. feature_importance.png
**What it shows**: Which features matter most

**Example features**:
1. **arm_swing** (0.18): Reduced arm movement in OFF state
2. **cadence** (0.15): Slower walking in OFF state
3. **acc_Ay_std** (0.12): Irregular vertical movement
4. **steps** (0.10): Fewer steps in OFF state
5. **acc_Ay_mean** (0.08): Lower movement intensity

**Interpretation**:
- Top features align with clinical knowledge
- If unexpected features dominate → investigate data issues
- Gait features (arm_swing, cadence) should be top 5

#### 4. confusion_matrix.png
**What it shows**: Prediction breakdown

**Example**:
```
              Predicted
           ON    OFF
Actual ON  345    22     (94% recall for ON)
       OFF  13   350     (96% recall for OFF)
```

**Interpretation**:
- **True Positives (350)**: Correctly detected OFF states
- **True Negatives (345)**: Correctly detected ON states
- **False Positives (13)**: Wrongly predicted OFF (actually ON)
- **False Negatives (22)**: Missed OFF states (predicted ON)

**Clinical Impact**:
- **False Negatives**: Missed OFF episodes → patient not warned
- **False Positives**: False alarms → patient takes unnecessary medication

**Goal**: Minimize false negatives (safety critical)

#### 5. roc_curve.png
**What it shows**: Trade-off between true positive and false positive rates

**Interpretation**:
- Curve close to top-left corner = Better model
- Area under curve (AUC) = Overall performance
- Can adjust threshold for different precision/recall balance

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: "No valid data found"
**Causes**:
- ON/ or OFF/ folders are empty
- CSV files missing required columns
- All files too short (<320 samples)

**Solutions**:
- Check folder paths
- Verify CSV format: must have Ax, Ay, Az, Gx, Gy, Gz columns
- Ensure files have >320 samples (10 seconds at 32 Hz)

#### Issue 2: Low Accuracy (<85%)
**Causes**:
- Mislabeled data (ON labeled as OFF or vice versa)
- Imbalanced dataset (90% ON, 10% OFF)
- Poor data quality (sensor errors, missing values)
- Patients in transitional state (not clearly ON or OFF)

**Solutions**:
- **Review labels**: Use clinical data to verify ON/OFF states
- **Balance dataset**: Aim for 40-60% of each class
- **Check data quality**: Plot some examples, look for anomalies
- **Exclude uncertain**: Only use clearly ON or clearly OFF recordings

#### Issue 3: High Training Accuracy, Low Test Accuracy (Overfitting)
**Example**: Training=99%, Test=85%

**Causes**:
- Model memorizing training data
- Too complex model (max_depth too high)
- Too few training examples

**Solutions**:
- Reduce `max_depth` (try 5 or 6)
- Increase `min_samples_leaf` (try 3 or 4)
- Get more training data
- Use fewer features (reduce `top_k`)

#### Issue 4: CV Accuracy Varies Widely Between Folds
**Example**: Fold accuracies: 95%, 88%, 92%, 85%, 90%

**Causes**:
- Small dataset
- Patient-specific patterns (model learns one patient, fails on others)
- Non-representative split

**Solutions**:
- Get more data from more patients
- Ensure cross-validation splits by patient (not randomly)
- Normalize features per patient

#### Issue 5: Training Too Slow
**Solutions**:
- Use `--fast` flag to skip plots
- Reduce `window_size_sec` (but keep ≥8 seconds)
- Use fewer CSV files for initial testing
- Reduce `n_estimators` to 50 (faster, slightly less accurate)

### Validation Checklist

Before trusting your model:

✅ **Test accuracy** >93%  
✅ **CV accuracy** close to test accuracy (±2%)  
✅ **AUC score** >0.95  
✅ **Precision and Recall** >90% for both classes  
✅ **Top features** make clinical sense (arm_swing, cadence, etc.)  
✅ **Confusion matrix** shows balanced performance  
✅ **No overfitting** (training accuracy ≈ test accuracy)  

---

## 📚 Next Steps

After training a good model:

1. **Export Model**:
   - Save trained model using `joblib` or `pickle`
   - Convert to ONNX format for production deployment

2. **Real-Time Detection**:
   - Integrate model into mobile app
   - Process live sensor streams
   - Provide instant OFF state warnings

3. **Clinical Validation**:
   - Test with new patients (never seen during training)
   - Compare predictions with clinical assessments
   - Collect feedback from patients and doctors

4. **Continuous Improvement**:
   - Collect more data from diverse patients
   - Retrain periodically with new data
   - Monitor performance in real-world use

---

## 📖 References

- MDS-UPDRS: Movement Disorder Society - Unified Parkinson's Disease Rating Scale
- Research papers on gait analysis and Parkinson's detection
- Clinical guidelines for ON/OFF state assessment

---

**Last Updated**: January 2026  
**Project**: aCare - Parkinson's Disease Monitoring System  
**Version**: 1.0