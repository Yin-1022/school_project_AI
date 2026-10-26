from time import monotonic

from .types import ExecutionState, RuntimeAction


class ExecutionController:
    def __init__(self):
        self.state = ExecutionState()

    def can_send(self) -> bool:
        return not self.state.busy

    def on_action_sent(self, action: RuntimeAction) -> None:
        if self.state.busy:
            raise RuntimeError("Cannot send action while busy")

        self.state.busy = True
        self.state.waiting_start = True
        self.state.current_action = action
        self.state.sent_at = monotonic()
        self.state.started_at = None

    def on_action_started(self) -> None:
        if self.state.busy or not self.state.waiting_start:
            self.state.waiting_start = False
            self.state.started_at = monotonic()

    def on_action_end(self) -> None:
        if not self.state.busy:
            return
        completed_action = self.state.current_action
        self.reset()
        return completed_action

    def reset(self) -> None:
        self.state = ExecutionState()