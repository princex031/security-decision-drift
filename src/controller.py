"""
Security Decision Drift
=======================

Controller implementation for the controlled decision-drift experiment.

Design principles
-----------------
1. Experiment parameters are supplied externally.
2. The controller does not silently choose missing experimental values.
3. Threat scoring, intervention attribution, and thresholding remain
   separately testable.
4. Reality Check applies the paper-specified intervention subtraction:

       score_t = threat(y_t) - lambda * intervention_explanation_t

5. Every decision can retain the intermediate quantities required for
   later audit and statistical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


class ConfigurationError(ValueError):
    """Raised when an experiment parameter is missing or invalid."""


@dataclass(frozen=True)
class ControllerConfig:
    """Immutable controller configuration."""

    lambda_value: Optional[float] = None
    threshold: Optional[float] = None

    def validate(self) -> None:
        if self.lambda_value is None:
            raise ConfigurationError(
                "lambda_value is not configured. "
                "Do not infer it from reported results."
            )

        if self.threshold is None:
            raise ConfigurationError(
                "threshold is not configured. "
                "Do not infer it from reported results."
            )

        if not isfinite(self.lambda_value):
            raise ConfigurationError("lambda_value must be finite.")

        if not isfinite(self.threshold):
            raise ConfigurationError("threshold must be finite.")

        if self.lambda_value < 0:
            raise ConfigurationError(
                "lambda_value must be non-negative."
            )


@dataclass(frozen=True)
class DecisionRecord:
    """Auditable representation of one controller decision."""

    observation: float
    threat_score: float
    intervention_explanation: float
    adjusted_score: float
    threshold: float
    escalated: bool


def threat_score(observation: float) -> float:
    """
    Convert an observation into the controller's threat score.

    The supplied paper describes the controller at the scalar
    score/threshold level; this function deliberately does not invent
    an additional classifier or learned model.
    """
    value = float(observation)

    if not isfinite(value):
        raise ValueError("observation must be finite.")

    return value


def reality_check_score(
    observation: float,
    intervention_explanation: float,
    lambda_value: float,
) -> float:
    """
    Apply the Reality Check adjustment.

    score_t = threat(y_t) - lambda * intervention_explanation_t
    """
    observation_value = float(observation)
    explanation_value = float(intervention_explanation)
    lambda_value = float(lambda_value)

    if not all(
        isfinite(value)
        for value in (
            observation_value,
            explanation_value,
            lambda_value,
        )
    ):
        raise ValueError("All controller inputs must be finite.")

    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative.")

    return (
        threat_score(observation_value)
        - lambda_value * explanation_value
    )


def make_decision(
    observation: float,
    intervention_explanation: float,
    config: ControllerConfig,
) -> DecisionRecord:
    """
    Produce one auditable Reality Check decision.
    """
    config.validate()

    raw_score = threat_score(observation)

    adjusted_score = reality_check_score(
        observation=observation,
        intervention_explanation=intervention_explanation,
        lambda_value=config.lambda_value,  # type: ignore[arg-type]
    )

    escalated = adjusted_score >= config.threshold  # type: ignore[operator]

    return DecisionRecord(
        observation=float(observation),
        threat_score=raw_score,
        intervention_explanation=float(intervention_explanation),
        adjusted_score=adjusted_score,
        threshold=float(config.threshold),  # type: ignore[arg-type]
        escalated=escalated,
    )