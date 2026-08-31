"""
Experiment orchestration.

Keeps paired-condition generation separate from the controller and
statistical analysis so that the causal comparison remains explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import ControllerConfig, DecisionRecord, make_decision
from .intervention import Intervention
from .reality_check import estimate_attribution, reality_check
from .simulator import generate_noise, make_observation


@dataclass(frozen=True)
class PairedCase:
    """One matched control/treatment case."""

    seed: int
    episode: int
    step: int
    external_state: float
    attack_signal: float
    noise: float
    control_observation: float
    treated_observation: float
    control_decision: DecisionRecord
    treated_decision: DecisionRecord


def run_paired_case(
    *,
    seed: int,
    episode: int,
    step: int,
    external_state: float,
    attack_signal: float,
    intervention_effect: float,
    controller: ControllerConfig,
) -> PairedCase:
    """
    Generate a matched pair.

    Both conditions receive the same external state and same stochastic
    noise. Only intervention history/effect differs.
    """

    controller.validate()

    rng = np.random.default_rng(seed)

    # Noise is generated once and shared by both matched conditions.
    noise = generate_noise(rng, variance=0.0)

    control = make_observation(
        external_state=external_state,
        attack_signal=attack_signal,
        intervention_effect=0.0,
        noise=noise,
    )

    intervention = Intervention(
        episode=episode,
        step=step,
        action="defensive_intervention",
        magnitude=intervention_effect,
        metadata={"paired": True},
    )

    treated = make_observation(
        external_state=external_state,
        attack_signal=attack_signal,
        intervention_effect=intervention_effect,
        noise=noise,
    )

    control_decision = make_decision(
        observation=control.value,
        intervention_explanation=0.0,
        config=controller,
    )

    attribution = estimate_attribution(
        observation=treated.value,
        interventions=[intervention],
        expected_effects=[intervention_effect],
    )

    treated_decision = reality_check(
        observation=treated.value,
        attribution=attribution,
        config=controller,
    )

    return PairedCase(
        seed=seed,
        episode=episode,
        step=step,
        external_state=external_state,
        attack_signal=attack_signal,
        noise=noise,
        control_observation=control.value,
        treated_observation=treated.value,
        control_decision=control_decision,
        treated_decision=treated_decision,
    )