
from collections.abc import Mapping

from runtime.types import RuntimeAction, ActionSource


class A3CAdapter:
    def __init__(self, action_values: Mapping[str, float]):
        self.action_values = dict(action_values)

    def to_runtime_action(self, action_name: str) -> RuntimeAction:
        if action_name not in self.action_values:
            raise ValueError(f"Invalid A3C action: {action_name}")

        return RuntimeAction(
            name=action_name,
            value=float(self.action_values[action_name]),
            source=ActionSource.POLICY,
        )

    def to_osc_payload(self, action: RuntimeAction) -> tuple:
        # A3C UE OSC contract: (action_name, value)
        return (action.name, float(action.value))
