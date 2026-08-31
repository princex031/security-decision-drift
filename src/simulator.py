"""
Controlled simulator for Security Decision Drift.

The simulator keeps the external state and stochastic noise explicit.
Intervention history is the manipulated variable in the paired test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One simulated security observation."""

    external_state: float
    attack_signal: float
    intervention_effect: float
    noise: float

    @property
    def value(self) -> float:
        """Observed scalar security signal."""
        return (
            self.external_state
            + self.attack_signal
            + self.intervention_effect
            + self.noise
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one simulator run."""

    episodes: int
    steps_per_episode: int
    noise_variance: Optional[float] = None

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive.")

        if self.steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be positive.")

        if self.noise_variance is None:
            raise ValueError(
                "noise_variance is not configured yet."
            )

        if self.noise_variance < 0:
            raise ValueError(
                "noise_variance must be non-negative."
            )


def generate_noise(
    rng: np.random.Generator,
    variance: float,
) -> float:
    """Generate one Gaussian noise realization."""

    if variance < 0:
        raise ValueError("variance must be non-negative.")

    return float(rng.normal(0.0, np.sqrt(variance)))


def make_observation(
    *,
    external_state: float,
    attack_signal: float,
    intervention_effect: float,
    noise: float,
) -> Observation:
    """Create one explicit observation record."""

    return Observation(
        external_state=float(external_state),
        attack_signal=float(attack_signal),
        intervention_effect=float(intervention_effect),
        noise=float(noise),
    )