# Predictive Maintenance – Evaluation Report

## 1. Objective

The objective is to predict whether an industrial machine is likely to experience a failure using its operating and sensor data.

This is a binary classification problem:

- 0: No machine failure
- 1: Machine failure

## 2. Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI)

- Total records: 10,000
- Total columns: 14
- Missing values: 0
- Duplicate rows: 0
- No failure: 9,661 (96.61%)
- Machine failure: 339 (3.39%)

The dataset is imbalanced because failure cases are much fewer than normal cases.

UDI and Product ID were excluded because they are identifiers. TWF, HDF, PWF, OSF and RNF were excluded because they describe failure modes and could cause target leakage.

## 3. Exploratory Data Analysis

The analysis included:

- Target class distribution
- Machine type distribution
- Sensor feature distributions
- Outlier detection using the IQR method
- Correlation heatmap

Outliers were detected in:

- Rotational speed: 418
- Torque: 69

These values were retained because they may represent genuine machine operating conditions.

Torque had the highest positive correlation with machine failure among the sensor features.

## 4. Preprocessing

- Selected six input features.
- Used median imputation for numerical features.
- Used most-frequent imputation for categorical features.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to machine type.
- Split the data into 80% training and 20% testing.
- Used stratified splitting to preserve class proportions.

## 5. Model Comparison

Three classification models were trained and evaluated.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression |   82.45% |    14.18% | 82.35% |   24.19% |  90.70% | 38.18% |
| Random Forest       |   98.00% |    93.75% | 44.12% |   60.00% |  96.61% | 78.02% |
| SVM                 |   91.75% |    27.85% | 89.71% |   42.51% |  96.55% | 61.19% |

PR-AUC was considered important because the dataset is highly imbalanced. Accuracy alone can be misleading when most machines do not fail.

## 6. Hyperparameter Tuning

RandomizedSearchCV was used to tune the Random Forest model.

- Cross-validation: 5-fold StratifiedKFold
- Number of parameter combinations: 20
- Scoring metric: Average Precision

Best parameters:

- n_estimators: 400
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: log2
- class_weight: balanced_subsample

Best cross-validation PR-AUC: 0.7708

### Tuned Model Results

- Precision: 90.70%
- Recall: 57.35%
- F1-score: 70.27%
- ROC-AUC: 96.15%
- PR-AUC: 80.99%

The tuned model improved recall, F1-score and PR-AUC compared with the baseline Random Forest.

## 7. Feature Importance

The top three features were:

1. Torque: 32.14%
2. Rotational speed: 29.92%
3. Tool wear: 21.29%

These features contributed most to the tuned Random Forest's predictions.

Feature importance indicates model contribution, not a causal relationship.

## 8. Error Analysis

Confusion matrix:

|                   | Predicted No Failure | Predicted Failure |
| ----------------- | -------------------: | ----------------: |
| Actual No Failure |                 1928 |                 4 |
| Actual Failure    |                   29 |                39 |

- True Positives: 39
- True Negatives: 1928
- False Positives: 4
- False Negatives: 29

False positives may result in unnecessary inspections or maintenance.

False negatives are more concerning because actual failures may go undetected.

## 9. Prediction on New Inputs

A separate Python script, predict_new.py, accepts machine sensor values and predicts whether the machine is likely to fail.

The script loads the saved preprocessing object and tuned Random Forest model to process new inputs and generate predictions.

## 10. Conclusion and Limitations

The tuned Random Forest achieved a PR-AUC of 80.99% and an F1-score of 70.27%.

It provides a useful starting point for predictive maintenance. However, it detected only 57.35% of actual failures on the test set.

Further improvements could include threshold tuning, additional feature engineering and evaluation on more real-world machine data.

The model's predictions should support maintenance decisions rather than replace expert inspection.
