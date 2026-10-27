from time import monotonic

from .types import ActionSource, ExecutionState, RuntimeAction, ExecutionResult, ExecutionEndReason


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

    def on_action_end(self) -> ExecutionResult | None:
        if not self.state.busy:
            return
        action = self.state.current_action

        if action is None:
            return None

        result = ExecutionResult(
            action=action,
            reason=ExecutionEndReason.COMPLETED,
        )

        self.reset()
        return result

    def check_timeout(self, now: float | None = None) -> ExecutionResult |None:
        if now is None:
            now = monotonic()

        if not self.state.busy:
            return None

        action = self.state.current_action
        if action is None:
            return None

        if self.state.waiting_start:
            if now - self.state.sent_at >= self.start_timeout:
                result = ExecutionResult(
                    action=action,
                    reason=ExecutionEndReason.START_TIMEOUT,
                )
                self.reset()
                return result

        else:
            if now - self.state.started_at >= self.end_timeout:
                result = ExecutionResult(
                    action=action,
                    reason=ExecutionEndReason.END_TIMEOUT,
                )
                self.reset()
                return result

        return None


    def abort(
        self,
        reason: ExecutionEndReason,
    ) -> ExecutionResult | None:
        if reason != ExecutionEndReason.REJECTED:
            raise ValueError("Abort reason must be REJECTED")

        if not self.state.busy:
            return None

        action = self.state.current_action
        if action is None:
            return None

        result = ExecutionResult(
            action=action,
            reason=ExecutionEndReason.REJECTED,
        )

        self.reset()
        return result

                
    def reset(self) -> None:
        self.state = ExecutionState()