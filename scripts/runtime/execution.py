from time import monotonic

from .types import ExecutionState, RuntimeAction


class ExecutionController:
    def __init__(self, start_timeout: float = 1.0, end_timeout: float = 5.0) -> None:
        self.state = ExecutionState()
        self.start_timeout = start_timeout
        self.end_timeout = end_timeout

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
        if not self.state.busy or not self.state.waiting_start:
            return

        self.state.waiting_start = False
        self.state.started_at = monotonic()

    def on_action_end(self) -> RuntimeAction | None:
        if not self.state.busy:
            return
        completed_action = self.state.current_action
        self.reset()
        return completed_action

    def check_timeout(self, now: float | None = None) -> str |None:
        if now is None:
            now = monotonic()

        if not self.state.busy:
            return None
        else:
            if self.state.waiting_start:
                if now - self.state.sent_at >= self.start_timeout:
                    self.reset()
                    return "actstart_timeout"
                else:
                    return None
            else:
                if now - self.state.started_at >= self.end_timeout:
                    self.reset()
                    return "actend_timeout"
                else:
                    return None
                
    def reset(self) -> None:
        self.state = ExecutionState()