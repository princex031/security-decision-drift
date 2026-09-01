from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


RESULTS = Path("results/unsw_nb15/reality_check_predictions.csv")


df = pd.read_csv(RESULTS)

y = df["label"]

baseline = df["baseline_decision"]
reality = df["reality_check_decision"]


print("=== BASELINE CONFUSION MATRIX ===")
print(confusion_matrix(y, baseline))

print("\n=== REALITY CHECK CONFUSION MATRIX ===")
print(confusion_matrix(y, reality))