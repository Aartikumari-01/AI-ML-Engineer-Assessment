
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("task1_ml/reports/figures", exist_ok=True)

model = joblib.load(
    "task1_ml/models/tuned_random_forest.joblib"
)

preprocessor = joblib.load(
    "task1_ml/models/preprocessor.joblib"
)

feature_names = preprocessor.get_feature_names_out()

importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    "task1_ml/reports/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    "task1_ml/reports/figures/feature_importance.png"
)

plt.close()

print("\nFeature importance graph saved successfully!")
print("Feature importance report saved successfully!")
