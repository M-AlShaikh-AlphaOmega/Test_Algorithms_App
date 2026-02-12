# Parkinson's ON/OFF Detection Results

## Dataset Summary
- Total samples: 24732939
- Duration: 111.53 seconds
- Windows: 1545807
- Features: 206

## Model Performance
- Cross-validation folds: 5
- CV accuracy (train): 0.6696 ± 0.0003
- Training samples: 1236645
- Test samples (held-out): 309162
- Held-out test accuracy: 0.6697
- Precision: 0.5830
- Recall: 0.6697
- F1-score: 0.5463

## Confusion Matrix
```
[[205749, 1931], [100195, 1287]]
```

## Classification Report
```
              precision    recall  f1-score   support

         OFF       0.67      0.99      0.80    207680
          ON       0.40      0.01      0.02    101482

    accuracy                           0.67    309162
   macro avg       0.54      0.50      0.41    309162
weighted avg       0.58      0.67      0.55    309162

```

## Top 10 Features
1. acc_ax_kurtosis: 0.0170
2. acc_ax_kurtosis_zscore: 0.0170
3. gyro_gy_kurtosis_zscore: 0.0168
4. gyro_gy_kurtosis: 0.0167
5. acc_ax_skewness_zscore: 0.0167
6. acc_ay_skewness_zscore: 0.0167
7. acc_az_skewness: 0.0167
8. acc_ay_skewness: 0.0167
9. acc_ax_skewness: 0.0167
10. acc_az_skewness_zscore: 0.0166
