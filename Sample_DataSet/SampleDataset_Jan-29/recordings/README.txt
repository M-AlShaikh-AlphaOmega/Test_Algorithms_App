================================================================================
HOW TO COLLECT REAL LABELED DATA FOR PARKINSON'S ON/OFF DETECTION
================================================================================

REQUIRED DATA:
--------------
1. Multiple recordings from Parkinson's patients
2. Each recording labeled as ON or OFF based on medication state

FILE NAMING CONVENTION:
-----------------------
- patient_001_OFF.csv  <- Recorded when medication has WORN OFF
- patient_001_ON.csv   <- Recorded when medication is EFFECTIVE
- patient_002_OFF.csv
- patient_002_ON.csv
...

CSV FORMAT:
-----------
Each file should have columns: time, X, Y, Z

Example:
time,X,Y,Z
0.0,-6.51,-2.88,6.63
0.04,-5.76,-3.31,6.57
0.08,-5.67,-1.85,6.91
...

WHEN TO RECORD:
---------------
OFF State (label=0):
- Record BEFORE patient takes medication (morning, before first dose)
- Or record 4-6 hours after last dose (when medication wears off)
- Patient may show: tremor, slow movement, stiffness

ON State (label=1):
- Record 30-60 minutes AFTER taking medication
- When patient feels medication is working
- Patient has: smooth movement, reduced tremor

RECOMMENDED AMOUNT:
-------------------
Minimum: 30 patients x 2 recordings (ON + OFF) = 60 files
Better:  50 patients x 2 recordings = 100 files
Best:    100+ patients with multiple sessions

RECORDING DURATION:
-------------------
- 10-30 seconds per recording
- Keep patient doing consistent activity (e.g., walking, sitting)

HOW TO USE:
-----------
Place your CSV files in this folder, then modify RandomForest-Train.py:

    # Instead of simulated data, use:
    X, y, feature_names = load_labeled_recordings('./recordings/')

================================================================================
