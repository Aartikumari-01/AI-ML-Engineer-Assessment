
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("task1_ml/data/ai4i2020.csv")

os.makedirs("task1_ml/reports/figures", exist_ok=True)

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("\n========== MACHINE TYPE DISTRIBUTION ==========")
print(df["Type"].value_counts())

print("\n========== OUTLIER ANALYSIS USING IQR ==========")

for col in sensor_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print("\nFeature:", col)
    print("Lower limit:", round(lower, 2))
    print("Upper limit:", round(upper, 2))
    print("Outliers:", len(outliers))

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[col])
    plt.title("Outlier Analysis - " + col)
    plt.tight_layout()
    plt.savefig(
        "task1_ml/reports/figures/" + col.replace(" ", "_") + "_boxplot.png"
    )
    plt.close()

print("\n========== FAILURE DISTRIBUTION GRAPH ==========")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Observations")
plt.tight_layout()
plt.savefig("task1_ml/reports/figures/failure_distribution.png")
plt.close()

print("Failure distribution graph saved.")

print("\n========== MACHINE TYPE GRAPH ==========")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Type")
plt.title("Machine Type Distribution")
plt.xlabel("Machine Type")
plt.ylabel("Number of Observations")
plt.tight_layout()
plt.savefig("task1_ml/reports/figures/machine_type.png")
plt.close()

print("Machine type graph saved.")

print("\n========== CORRELATION ANALYSIS ==========")

correlation_cols = sensor_cols + ["Machine failure"]
correlation = df[correlation_cols].corr()

print(correlation["Machine failure"].sort_values(ascending=False))

plt.figure(figsize=(9, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("task1_ml/reports/figures/correlation_heatmap.png")
plt.close()

print("\nAll EDA graphs saved successfully!")
