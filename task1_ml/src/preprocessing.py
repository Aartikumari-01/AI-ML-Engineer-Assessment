
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_and_split_data():

    df = pd.read_csv("task1_ml/data/ai4i2020.csv")

    target = "Machine failure"

    features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[features]
    y = df[target]

    categorical_features = ["Type"]

    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\n========== DATA SPLIT ==========")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\n========== TARGET DISTRIBUTION ==========")
    print("Training:")
    print(y_train.value_counts())

    print("\nTesting:")
    print(y_test.value_counts())

    print("\n========== PROCESSED DATA SHAPE ==========")
    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    load_and_split_data()
