from runtime.types import RuntimeAction, ActionSource
from constant import ACTION_ID_TO_NAME

ACTION_SPACE: frozenset[str] = frozenset(ACTION_ID_TO_NAME.values())

def to_runtime_action(action_name: str) -> RuntimeAction:
    if action_name not in ACTION_SPACE:
        raise ValueError(f"Invalid action name: {action_name}")

    if action_name == "SearchTurnLeft":
        return RuntimeAction(
            name="SearchTurn",
            value=-50.0,
            source=ActionSource.POLICY,
        )
    elif action_name == "SearchTurnRight":
        return RuntimeAction(
            name="SearchTurn",
            value=50.0,
            source=ActionSource.POLICY,
        )
    elif action_name == "PatrolStepLeft":
        return RuntimeAction(
            name="PatrolStep",
            value=-50.0,
            source=ActionSource.POLICY,
        )
    elif action_name == "PatrolStepRight":
        return RuntimeAction(
            name="PatrolStep",
            value=50.0,
            source=ActionSource.POLICY,
        )
    else:
        return RuntimeAction(
            name=action_name,
            value=0.0,
            source=ActionSource.POLICY,
        )

def to_osc_payload(action: RuntimeAction) -> dict:
    if action.name == "SearchTurn" or action.name == "PatrolStep":
        return {
            "action": action.name,
            "angle": action.value,
        }
    else:
        return {
            "action": action.name,
        }