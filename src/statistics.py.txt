"""
Statistical utilities for the research experiments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Return mean, lower CI, and upper CI using bootstrap resampling.
    """

    data = np.asarray(values, dtype=float)

    if data.size == 0:
        raise ValueError("values must not be empty.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if resamples <= 0:
        raise ValueError("resamples must be positive.")

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        data,
        size=(resamples, data.size),
        replace=True,
    )

    means = samples.mean(axis=1)

    alpha = 1.0 - confidence

    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))

    return float(data.mean()), lower, upper