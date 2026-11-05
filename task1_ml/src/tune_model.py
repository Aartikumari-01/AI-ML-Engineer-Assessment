
import os
import joblib

from preprocessing import load_and_split_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


os.makedirs("task1_ml/models", exist_ok=True)
os.makedirs("task1_ml/reports", exist_ok=True)

X_train, X_test, y_train, y_test, preprocessor = load_and_split_data()

random_forest = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

parameter_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"]
}

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=parameter_grid,
    n_iter=20,
    scoring="average_precision",
    cv=cross_validation,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n========== STARTING HYPERPARAMETER TUNING ==========")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n========== BEST PARAMETERS ==========")
print(search.best_params_)

print("\nBest Cross-Validation PR-AUC:", round(search.best_score_, 4))

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

print("\n========== TUNED MODEL TEST RESULTS ==========")
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("PR-AUC:", round(average_precision_score(y_test, probabilities), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(
    best_model,
    "task1_ml/models/tuned_random_forest.joblib"
)

print("\nTuned Random Forest model saved successfully!")
