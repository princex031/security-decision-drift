"""
Research experiment runner.

Runs matched observations through the three conceptual conditions:
Baseline, History-aware, and Reality Check.

This module does not invent missing paper parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .controller import ControllerConfig, DecisionRecord, make_decision
from .intervention import Intervention
from .reality_check import estimate_attribution, reality_check


@dataclass(frozen=True)
class ConditionResult:
    """Decision outputs for one matched observation."""

    baseline: DecisionRecord
    history_aware: DecisionRecord
    reality_check: DecisionRecord


def run_conditions(
    observation: float,
    intervention: Intervention | None,
    config: ControllerConfig,
) -> ConditionResult:
    """
    Evaluate the same observation under all three conditions.
    """

    baseline = make_decision(
        observation=observation,
        intervention_explanation=0.0,
        config=config,
    )

    if intervention is None:
        history_explanation = 0.0
        attribution = estimate_attribution(
            observation=observation,
            interventions=[],
        )
    else:
        history_explanation = abs(float(intervention.magnitude))
        attribution = estimate_attribution(
            observation=observation,
            interventions=[intervention],
            expected_effects=[history_explanation],
        )

    history_aware = make_decision(
        observation=observation,
        intervention_explanation=history_explanation,
        config=config,
    )

    checked = reality_check(
        observation=observation,
        attribution=attribution,
        config=config,
    )

    return ConditionResult(
        baseline=baseline,
        history_aware=history_aware,
        reality_check=checked,
    )