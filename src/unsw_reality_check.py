from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/unsw_nb15")


def load_data():
    train = pd.read_csv(
        DATA_DIR / "UNSW_NB15_training-set.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "UNSW_NB15_testing-set.csv"
    )

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


def run_reality_check(model, test):
    drop_columns = ["id", "attack_cat", "label"]

    X_test = test.drop(columns=drop_columns)
    y_test = test["label"].to_numpy()

    probabilities = model.predict_proba(X_test)[:, 1]

    baseline_decision = (probabilities >= 0.5).astype(int)

    # Controlled intervention-history simulation.
    # This does NOT claim that UNSW-NB15 contains intervention history.
    intervention_effect = np.where(
        baseline_decision == 1,
        0.10,
        0.0,
    )

    adjusted_probability = np.clip(
        probabilities - intervention_effect,
        0.0,
        1.0,
    )

    reality_check_decision = (
        adjusted_probability >= 0.5
    ).astype(int)

    drift = (
        baseline_decision != reality_check_decision
    ).astype(int)

    results = pd.DataFrame(
        {
            "label": y_test,
            "baseline_probability": probabilities,
            "baseline_decision": baseline_decision,
            "adjusted_probability": adjusted_probability,
            "reality_check_decision": reality_check_decision,
            "decision_drift": drift,
        }
    )

    return results


if __name__ == "__main__":
    train, test = load_data()

    print("Loading UNSW-NB15...")
    print("Training:", train.shape)
    print("Testing :", test.shape)

    print("\nTraining baseline model...")
    model = build_model(train)

    print("Running Reality Check...")
    results = run_reality_check(model, test)

    drift_rate = results["decision_drift"].mean()

    baseline_false_actions = (
        (results["baseline_decision"] == 1)
        & (results["label"] == 0)
    ).sum()

    reality_false_actions = (
        (results["reality_check_decision"] == 1)
        & (results["label"] == 0)
    ).sum()

    print("\n=== REALITY CHECK — UNSW-NB15 ===")

    print(
        "Baseline false actions:",
        baseline_false_actions,
    )

    print(
        "Reality Check false actions:",
        reality_false_actions,
    )

    print(
        "Decision drift rate:",
        drift_rate,
    )

    output_path = (
        Path("results/unsw_nb15")
        / "reality_check_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:", output_path)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
baseline_accuracy = accuracy_score(
    results["label"],
    results["baseline_decision"],
)

baseline_precision = precision_score(
    results["label"],
    results["baseline_decision"],
)

baseline_recall = recall_score(
    results["label"],
    results["baseline_decision"],
)

baseline_f1 = f1_score(
    results["label"],
    results["baseline_decision"],
)


reality_accuracy = accuracy_score(
    results["label"],
    results["reality_check_decision"],
)

reality_precision = precision_score(
    results["label"],
    results["reality_check_decision"],
)

reality_recall = recall_score(
    results["label"],
    results["reality_check_decision"],
)

reality_f1 = f1_score(
    results["label"],
    results["reality_check_decision"],
)
print("\n=== BASELINE METRICS ===")
print("Accuracy :", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall   :", baseline_recall)
print("F1       :", baseline_f1)

print("\n=== REALITY CHECK METRICS ===")
print("Accuracy :", reality_accuracy)
print("Precision:", reality_precision)
print("Recall   :", reality_recall)
print("F1       :", reality_f1)