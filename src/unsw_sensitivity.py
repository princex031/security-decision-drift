from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(
    "results/unsw_nb15/reality_check_predictions.csv"
)

df = pd.read_csv(RESULTS)


for effect in [0.05, 0.10, 0.15]:
    adjusted = np.where(
        df["baseline_decision"] == 1,
        np.clip(
            df["baseline_probability"] - effect,
            0.0,
            1.0,
        ),
        df["baseline_probability"],
    )

    decision = (adjusted >= 0.5).astype(int)

    false_actions = (
        (decision == 1)
        & (df["label"] == 0)
    ).sum()

    drift = (
        decision
        != df["baseline_decision"]
    ).mean()

    print(
        f"Effect={effect:.2f} | "
        f"False actions={false_actions} | "
        f"Drift={drift:.6f}"
    )