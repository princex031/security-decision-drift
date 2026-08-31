"""
Reality Check layer for Security Decision Drift.

The Reality Check does not replace the underlying threat detector.
It uses recent intervention context to estimate whether part of a
new observation can be explained by an earlier defensive action.

The implementation is intentionally lightweight so that the experiment
tests the intervention-history mechanism rather than a particular AI
model or classifier family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .intervention import Intervention, intervention_id
from .controller import (
    ControllerConfig,
    DecisionRecord,
    make_decision,
)


@dataclass(frozen=True)
class AttributionResult:
    """
    Attribution summary for one observation.

    `explained_fraction` is a bounded experimental quantity. It is not
    presented as a universal causal probability.
    """

    explained_fraction: float
    matched_interventions: tuple[str, ...]
    confidence: float


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value to a closed interval."""

    return max(lower, min(upper, float(value)))


def estimate_attribution(
    observation: float,
    interventions: Iterable[Intervention],
    expected_effects: Optional[Iterable[float]] = None,
) -> AttributionResult:
    """
    Estimate how much of an observation is explainable by recent
    intervention effects.

    This function deliberately keeps the attribution rule simple.
    The actual intervention-effect model remains an experimental
    configuration rather than hidden inside the monitor.
    """

    observation_value = float(observation)

    if not isinstance(observation_value, (int, float)):
        raise TypeError("observation must be numeric.")

    intervention_list = list(interventions)

    if expected_effects is None:
        effects = [abs(float(item.magnitude)) for item in intervention_list]
    else:
        effects = [abs(float(value)) for value in expected_effects]

    if len(effects) != len(intervention_list):
        raise ValueError(
            "expected_effects must match the number of interventions."
        )

    if not intervention_list:
        return AttributionResult(
            explained_fraction=0.0,
            matched_interventions=(),
            confidence=1.0,
        )

    total_explained = sum(effects)

    if abs(observation_value) < 1e-12:
        explained_fraction = 0.0
    else:
        explained_fraction = total_explained / abs(observation_value)

    explained_fraction = _clamp(explained_fraction)

    matched = tuple(
        intervention_id(item)
        for item, effect in zip(intervention_list, effects)
        if effect > 0
    )

    # Confidence is intentionally conservative: more contributing
    # interventions do not automatically mean higher confidence.
    confidence = 1.0 / (1.0 + max(0, len(intervention_list) - 1))

    return AttributionResult(
        explained_fraction=explained_fraction,
        matched_interventions=matched,
        confidence=confidence,
    )


def reality_check(
    observation: float,
    attribution: AttributionResult,
    config: ControllerConfig,
) -> DecisionRecord:
    """
    Run the Reality Check decision for one observation.

    The attribution result is converted into the intervention-explanation
    term used by the controller's scoring rule.
    """

    explained = _clamp(attribution.explained_fraction)

    return make_decision(
        observation=observation,
        intervention_explanation=explained,
        config=config,
    )