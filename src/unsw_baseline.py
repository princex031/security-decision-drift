from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/unsw_nb15")


def load_data():
    train = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(DATA_DIR / "UNSW_NB15_testing-set.csv")

    return train, test


def build_model(train):
    drop_columns = ["id", "attack_cat", "label"]

    X_train = train.drop(columns=drop_columns)
    y_train = train["label"]

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, test):
    drop_columns = ["id", "attack_cat", "label"]

    X_test = test.drop(columns=drop_columns)
    y_test = test["label"]

    predictions = model.predict(X_test)

    print("\n=== UNSW-NB15 BASELINE ===")
    print("Accuracy :", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall   :", recall_score(y_test, predictions))
    print("F1       :", f1_score(y_test, predictions))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    train, test = load_data()

    print("Training:", train.shape)
    print("Testing :", test.shape)

    model = build_model(train)

    evaluate(model, test)