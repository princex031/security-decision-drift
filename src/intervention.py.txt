"""
Intervention tracking for Security Decision Drift.

The intervention layer records what the defender changed in the
environment so that downstream observations can be evaluated for
possible self-generated evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Intervention:
    """A single defensive intervention."""

    episode: int
    step: int
    action: str
    magnitude: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class InterventionEffect:
    """Estimated contribution of an intervention to an observation."""

    intervention_id: str
    explanation: float


def validate_intervention(intervention: Intervention) -> None:
    """Validate the basic integrity of an intervention record."""

    if intervention.episode < 0:
        raise ValueError("episode must be non-negative.")

    if intervention.step < 0:
        raise ValueError("step must be non-negative.")

    if not intervention.action:
        raise ValueError("action must not be empty.")


def intervention_id(intervention: Intervention) -> str:
    """Return a stable identifier for an intervention."""

    validate_intervention(intervention)

    return (
        f"episode-{intervention.episode}:"
        f"step-{intervention.step}:"
        f"{intervention.action}"
    )


def explain_observation(
    intervention: Intervention,
    effect: float,
) -> InterventionEffect:
    """
    Associate an observation-level explanation with an intervention.

    The effect value is supplied by the experiment/simulator. This
    function does not infer or invent an intervention-effect model.
    """

    validate_intervention(intervention)

    if not isinstance(effect, (int, float)):
        raise TypeError("effect must be numeric.")

    return InterventionEffect(
        intervention_id=intervention_id(intervention),
        explanation=float(effect),
    )