from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/unsw_nb15")


def load_unsw_nb15():
    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


if __name__ == "__main__":
    train, test = load_unsw_nb15()

    print("Training shape:", train.shape)
    print("Testing shape:", test.shape)

    print("\nColumns:")
    print(train.columns.tolist())

    if "label" in train.columns:
        print("\nTraining labels:")
        print(train["label"].value_counts())