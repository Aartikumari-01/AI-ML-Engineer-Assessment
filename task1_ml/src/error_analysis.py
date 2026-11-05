
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.makedirs("task1_ml/reports", exist_ok=True)

df = pd.read_csv("task1_ml/data/ai4i2020.csv")

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = joblib.load(
    "task1_ml/models/preprocessor.joblib"
)

model = joblib.load(
    "task1_ml/models/tuned_random_forest.joblib"
)

X_test_processed = preprocessor.transform(X_test)

predictions = model.predict(X_test_processed)
probabilities = model.predict_proba(X_test_processed)[:, 1]

results = X_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = predictions
results["Failure_Probability"] = probabilities

false_positives = results[
    (results["Actual"] == 0) &
    (results["Predicted"] == 1)
]

false_negatives = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 0)
]

true_positives = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 1)
]

true_negatives = results[
    (results["Actual"] == 0) &
    (results["Predicted"] == 0)
]

print("\n========== ERROR ANALYSIS ==========")

print("True Positives:", len(true_positives))
print("True Negatives:", len(true_negatives))
print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

print("\n========== FALSE POSITIVE EXAMPLES ==========")
print(false_positives.head(5).to_string(index=False))

print("\n========== FALSE NEGATIVE EXAMPLES ==========")
print(false_negatives.head(5).to_string(index=False))

false_positives.to_csv(
    "task1_ml/reports/false_positives.csv",
    index=False
)

false_negatives.to_csv(
    "task1_ml/reports/false_negatives.csv",
    index=False
)

results.to_csv(
    "task1_ml/reports/all_test_predictions.csv",
    index=False
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nError analysis reports saved successfully!")
