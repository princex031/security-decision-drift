from src.controller import ControllerConfig
from src.intervention import Intervention
from src.runner import run_conditions


def test_three_conditions_are_returned():
    config = ControllerConfig(
        lambda_value=1.0,
        threshold=0.5,
    )

    intervention = Intervention(
        episode=0,
        step=0,
        action="test",
        magnitude=0.1,
        metadata={},
    )

    result = run_conditions(
        observation=0.8,
        intervention=intervention,
        config=config,
    )

    assert result.baseline is not None
    assert result.history_aware is not None
    assert result.reality_check is not None