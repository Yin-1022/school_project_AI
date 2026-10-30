from .types import RuntimeAction, ActionSource


class ActionFallbackController:
    def __init__(self, fallback_value: float = 0.0) -> None:
        self.fallback_value = fallback_value
        self.pending = False
        self.rejected_action: RuntimeAction | None = None

    def on_notseeplayer(
        self,
        current_action: RuntimeAction | None,
    ) -> bool:
        if current_action is None or self.pending:
            return False

        if (
            current_action.source != ActionSource.POLICY
            or current_action.name not in {"StrafeLeft", "StrafeRight"}
        ):
            return False

        self.pending = True
        self.rejected_action = current_action
        return True

    def next_action(self) -> RuntimeAction | None:
        if not self.pending:
            return None

        return RuntimeAction(
            name="Advance",
            value=self.fallback_value,
            source=ActionSource.RUNTIME_FALLBACK,
        )

    def on_action_sent(self, action: RuntimeAction) -> None:
        if self.pending and action == self.next_action():
            self.pending = False
            self.rejected_action = None

    def cancel_pending(self) -> None:
        self.pending = False
        self.rejected_action = None

    def reset(self) -> None:
        self.pending = False
        self.rejected_action = None