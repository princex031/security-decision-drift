"""
Metrics used by the Security Decision Drift experiments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def decision_drift_rate(
    control_actions: Sequence[int],
    intervention_actions: Sequence[int],
) -> float:
    """
    Fraction of matched cases whose action class changes.
    """

    control = np.asarray(control_actions)
    treated = np.asarray(intervention_actions)

    if control.shape != treated.shape:
        raise ValueError(
            "control_actions and intervention_actions "
            "must have the same shape."
        )

    if control.size == 0:
        raise ValueError("At least one paired decision is required.")

    return float(np.mean(control != treated))


def false_action_rate(
    actions: Sequence[int],
    attack_flags: Sequence[int],
) -> float:
    """
    Fraction of non-attack steps receiving a non-zero response.
    """

    actions_array = np.asarray(actions)
    attack_array = np.asarray(attack_flags)

    if actions_array.shape != attack_array.shape:
        raise ValueError("Input arrays must have equal shape.")

    non_attack = attack_array == 0

    if not np.any(non_attack):
        return 0.0

    return float(
        np.mean(actions_array[non_attack] != 0)
    )


def attack_recall(
    actions: Sequence[int],
    attack_flags: Sequence[int],
) -> float:
    """
    Fraction of attack steps receiving a non-zero response.
    """

    actions_array = np.asarray(actions)
    attack_array = np.asarray(attack_flags)

    if actions_array.shape != attack_array.shape:
        raise ValueError("Input arrays must have equal shape.")

    attack = attack_array == 1

    if not np.any(attack):
        return 0.0

    return float(
        np.mean(actions_array[attack] != 0)
    )