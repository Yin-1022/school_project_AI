
import pytest

from runtime.controller import RuntimeController
from runtime.types import ActionSource, RuntimeAction


A3C_VALUES = {
    "Advance": 1.0,
    "StrafeLeft": 1.0,
    "StrafeRight": 1.0,
    "EvadeBack": 1.0,
    "Retreat": 1.0,
    "SearchTurn": 45.0,
    "PatrolStep": 30.0,
}


def test_impala_adapter():
    runtime = RuntimeController(model="IMPALA")

    action = runtime.policy_action("SearchTurnRight")

    assert action.name == "SearchTurn"
    assert action.value == 50.0
    assert action.source == ActionSource.POLICY
    assert runtime.osc_payload(action) == {
        "action": "SearchTurn",
        "angle": 50.0,
    }


def test_a3c_adapter():
    runtime = RuntimeController(
        model="A3C",
        action_values=A3C_VALUES,
    )

    action = runtime.policy_action("SearchTurn")

    assert action.name == "SearchTurn"
    assert action.value == 45.0
    assert action.source == ActionSource.POLICY
    assert runtime.osc_payload(action) == ("SearchTurn", 45.0)


@pytest.mark.parametrize("action_name,value", A3C_VALUES.items())
def test_a3c_action_values(action_name, value):
    runtime = RuntimeController(
        model="A3C",
        action_values=A3C_VALUES,
    )

    action = runtime.policy_action(action_name)

    assert action.name == action_name
    assert action.value == value


def test_runtime_action_preserves_override_value():
    runtime = RuntimeController(
        model="A3C",
        action_values=A3C_VALUES,
    )

    action = RuntimeAction(
        "SearchTurn",
        135.0,
        ActionSource.RUNTIME_HIT_REACTION,
    )

    assert runtime.osc_payload(action) == (
        "SearchTurn",
        135.0,
    )


def test_invalid_model():
    with pytest.raises(ValueError, match="Unsupported"):
        RuntimeController(model="DQN")


def test_invalid_a3c_action():
    runtime = RuntimeController(
        model="A3C",
        action_values=A3C_VALUES,
    )

    with pytest.raises(ValueError, match="Invalid"):
        runtime.policy_action("Hold")


def test_runtime_instances_are_independent():
    impala = RuntimeController(model="IMPALA")
    a3c = RuntimeController(
        model="A3C",
        action_values=A3C_VALUES,
    )

    action = a3c.policy_action("Advance")
    a3c.on_action_sent(action)

    assert not a3c.can_send()
    assert impala.can_send()

    a3c.reset()

    assert a3c.can_send()
