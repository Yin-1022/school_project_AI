
import pytest

from constant import ACTION_ID_TO_NAME
from runtime.types import RuntimeAction, ActionSource
from runtime.adapters.impala_adapter import (
    to_runtime_action,
    to_osc_payload,
)


# ============================================
# 1. IMPALA Policy Action Mapping
# ============================================

@pytest.mark.parametrize(
    "action_name, expected_name, expected_value",
    [
        ("Hold", "Hold", 0.0),
        ("Advance", "Advance", 0.0),
        ("StrafeLeft", "StrafeLeft", 0.0),
        ("StrafeRight", "StrafeRight", 0.0),
        ("EvadeBack", "EvadeBack", 0.0),
        ("Retreat", "Retreat", 0.0),
        ("SearchTurnLeft", "SearchTurn", -50.0),
        ("SearchTurnRight", "SearchTurn", 50.0),
        ("PatrolStepLeft", "PatrolStep", -50.0),
        ("PatrolStepRight", "PatrolStep", 50.0),
    ],
)
def test_policy_action_mapping(
    action_name,
    expected_name,
    expected_value,
):
    action = to_runtime_action(action_name)

    assert isinstance(action, RuntimeAction)
    assert action.name == expected_name
    assert action.value == expected_value
    assert action.source == ActionSource.POLICY


def test_all_policy_actions_covered():
    expected_actions = {
        "Hold",
        "Advance",
        "StrafeLeft",
        "StrafeRight",
        "EvadeBack",
        "Retreat",
        "SearchTurnLeft",
        "SearchTurnRight",
        "PatrolStepLeft",
        "PatrolStepRight",
    }

    assert set(ACTION_ID_TO_NAME.values()) == expected_actions

    for action_name in ACTION_ID_TO_NAME.values():
        action = to_runtime_action(action_name)
        assert action.source == ActionSource.POLICY


# ============================================
# 2. Invalid Policy Actions
# ============================================

@pytest.mark.parametrize(
    "invalid_name",
    [
        "Attack1",
        "SearchTurn",
        "PatrolStep",
        "UnknownAction",
        "",
        "advance",
        "SearchTurnLEFT",
    ],
)
def test_invalid_policy_action_raises_value_error(invalid_name):
    with pytest.raises(ValueError, match="Invalid action name"):
        to_runtime_action(invalid_name)


# ============================================
# 3. OSC Payload Conversion
# ============================================

@pytest.mark.parametrize(
    "action, expected_payload",
    [
        (
            RuntimeAction("Advance", 0.0, ActionSource.POLICY),
            {"action": "Advance"},
        ),
        (
            RuntimeAction("StrafeLeft", 0.0, ActionSource.POLICY),
            {"action": "StrafeLeft"},
        ),
        (
            RuntimeAction("SearchTurn", -50.0, ActionSource.POLICY),
            {"action": "SearchTurn", "angle": -50.0},
        ),
        (
            RuntimeAction("SearchTurn", 50.0, ActionSource.POLICY),
            {"action": "SearchTurn", "angle": 50.0},
        ),
        (
            RuntimeAction("PatrolStep", -50.0, ActionSource.POLICY),
            {"action": "PatrolStep", "angle": -50.0},
        ),
        (
            RuntimeAction("PatrolStep", 50.0, ActionSource.POLICY),
            {"action": "PatrolStep", "angle": 50.0},
        ),
        (
            RuntimeAction(
                "SearchTurn", 90.0, ActionSource.RUNTIME_RECOVERY
            ),
            {"action": "SearchTurn", "angle": 90.0},
        ),
        (
            RuntimeAction(
                "SearchTurn", -90.0, ActionSource.RUNTIME_RECOVERY
            ),
            {"action": "SearchTurn", "angle": -90.0},
        ),
        (
            RuntimeAction(
                "SearchTurn", 135.0, ActionSource.RUNTIME_HIT_REACTION
            ),
            {"action": "SearchTurn", "angle": 135.0},
        ),
        (
            RuntimeAction(
                "EvadeBack", 0.0, ActionSource.RUNTIME_RECOVERY
            ),
            {"action": "EvadeBack"},
        ),
        (
            RuntimeAction(
                "Advance", 0.0, ActionSource.RUNTIME_FALLBACK
            ),
            {"action": "Advance"},
        ),
    ],
)
def test_to_osc_payload(action, expected_payload):
    payload = to_osc_payload(action)

    assert payload == expected_payload


# ============================================
# 4. End-to-End Adapter Conversion
# ============================================

@pytest.mark.parametrize(
    "policy_name, expected_payload",
    [
        ("Hold", {"action": "Hold"}),
        ("Advance", {"action": "Advance"}),
        ("StrafeLeft", {"action": "StrafeLeft"}),
        ("StrafeRight", {"action": "StrafeRight"}),
        ("EvadeBack", {"action": "EvadeBack"}),
        ("Retreat", {"action": "Retreat"}),
        ("SearchTurnLeft", {"action": "SearchTurn", "angle": -50.0}),
        ("SearchTurnRight", {"action": "SearchTurn", "angle": 50.0}),
        ("PatrolStepLeft", {"action": "PatrolStep", "angle": -50.0}),
        ("PatrolStepRight", {"action": "PatrolStep", "angle": 50.0}),
    ],
)
def test_policy_to_osc_end_to_end(policy_name, expected_payload):
    runtime_action = to_runtime_action(policy_name)
    payload = to_osc_payload(runtime_action)

    assert runtime_action.source == ActionSource.POLICY
    assert payload == expected_payload


# ============================================
# 5. No Mutation
# ============================================

def test_osc_conversion_does_not_modify_action():
    action = RuntimeAction(
        name="SearchTurn",
        value=135.0,
        source=ActionSource.RUNTIME_HIT_REACTION,
    )

    original = action
    payload = to_osc_payload(action)

    assert action == original
    assert action.name == "SearchTurn"
    assert action.value == 135.0
    assert action.source == ActionSource.RUNTIME_HIT_REACTION
    assert payload["angle"] == 135.0
