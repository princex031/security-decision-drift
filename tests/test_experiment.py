import pytest

from src.controller import ControllerConfig
from src.experiment import run_paired_case


def test_paired_case_requires_valid_controller():
    config = ControllerConfig(
        lambda_value=1.0,
        threshold=0.5,
    )

    result = run_paired_case(
        seed=42,
        episode=0,
        step=0,
        external_state=0.2,
        attack_signal=0.8,
        intervention_effect=0.1,
        controller=config,
    )

    assert result.control_observation != result.treated_observation
    assert result.control_decision.threshold == 0.5
    assert result.treated_decision.threshold == 0.5