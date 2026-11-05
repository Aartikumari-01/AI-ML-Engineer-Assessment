
import pandas as pd

df = pd.read_csv("task1_ml/data/ai4i2020.csv")

print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MACHINE FAILURE DISTRIBUTION ==========")
print(df["Machine failure"].value_counts())

print("\n========== MACHINE FAILURE PERCENTAGE ==========")
print(df["Machine failure"].value_counts(normalize=True) * 100)

print("\n========== FAILURE MODE DISTRIBUTION ==========")
print(df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum())
