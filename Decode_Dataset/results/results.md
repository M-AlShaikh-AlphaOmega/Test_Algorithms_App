# Parkinson's ON/OFF Detection Results

## Dataset Summary
- Total samples: 5927687
- Duration: 111.53 seconds
- Windows: 370479
- Features: 206

## Model Performance
- Cross-validation folds: 5
- CV accuracy (train): 0.6692 ± 0.0005
- Training samples: 296383
- Test samples (held-out): 74096
- Held-out test accuracy: 0.6692
- Precision: 0.5945
- Recall: 0.6692
- F1-score: 0.5589

## Confusion Matrix
```
[[48738, 1133], [23380, 845]]
```

## Classification Report
```
              precision    recall  f1-score   support

         OFF       0.68      0.98      0.80     49871
          ON       0.43      0.03      0.06     24225

    accuracy                           0.67     74096
   macro avg       0.55      0.51      0.43     74096
weighted avg       0.59      0.67      0.56     74096

```

## Top 10 Features
1. acc_ax_skewness_zscore: 0.0171
2. acc_ax_skewness: 0.0170
3. acc_az_skewness: 0.0169
4. acc_ax_kurtosis_zscore: 0.0169
5. acc_ax_kurtosis: 0.0168
6. acc_az_skewness_zscore: 0.0168
7. acc_ay_kurtosis_zscore: 0.0167
8. acc_ay_skewness_zscore: 0.0167
9. acc_ay_skewness: 0.0167
10. acc_ay_spectral_entropy_zscore: 0.0166
