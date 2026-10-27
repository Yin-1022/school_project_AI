from .types import RuntimeAction, ActionSource


class HitReactionController:
    def __init__(
        self,
        turn_angle: float = 135.0,
        max_attempts: int = 2,
    ):
        self.turn_angle = turn_angle
        self.max_attempts = max_attempts
        self.attempts = 0
        self.pending = False

    def on_visible_update(self, visible: bool) -> None:
        if visible:
            self.pending = False
            self.attempts = 0

    def on_boss_hit(
        self,
        visible: bool,
        recovery_active: bool,
    ) -> None:
        if (
            not visible
            and not recovery_active
            and self.attempts < self.max_attempts
        ):
            self.pending = True

    def next_action(self) -> RuntimeAction | None:
        if self.pending:
            return RuntimeAction(
                "SearchTurn",
                self.turn_angle,
                ActionSource.RUNTIME_HIT_REACTION,
            )
        else:
            return None

    def on_action_sent(self, action: RuntimeAction) -> None:
        expected_action = self.next_action()

        if expected_action is None or action != expected_action:
            return

        self.pending = False
        self.attempts += 1

    def reset(self) -> None:
        self.attempts = 0
        self.pending = False