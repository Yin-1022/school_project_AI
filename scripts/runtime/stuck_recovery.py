from enum import Enum
from .types import RuntimeAction, ActionSource

class RecoveryStage(str, Enum):
    IDLE = "idle"
    TURN_PENDING = "turn_pending"
    TURN_EXECUTING = "turn_executing"
    EVADE_PENDING = "evade_pending"
    EVADE_EXECUTING = "evade_executing"

class StuckRecoveryController:
    def __init__(self) -> None:
        self.stage = RecoveryStage.IDLE
        self.turn_sign = 1
        self.turn_failed = False
        self.verify_pending = False
        self.verify_action: str | None = None

    def on_cantmove(self, action: RuntimeAction | None) -> None:            
        if (
            self.stage == RecoveryStage.TURN_EXECUTING
            and action is not None
            and action.name == "SearchTurn"
            and action.source == ActionSource.RUNTIME_RECOVERY
        ):
            self.turn_failed = True
            return

        if self.verify_pending and action is not None:
            if(
                self.stage == RecoveryStage.IDLE
                and self.verify_action is not None
                and action.source == ActionSource.POLICY
                and action.name == self.verify_action
            ):
                self.verify_pending = False
                self.verify_action = None
                self.stage = RecoveryStage.EVADE_PENDING
                return

        if self.stage == RecoveryStage.IDLE:
            self.verify_pending = False
            self.verify_action = None
            self.stage = RecoveryStage.TURN_PENDING
            self.turn_failed = False
        

    def next_action(self) -> RuntimeAction | None:
        if self.stage == RecoveryStage.TURN_PENDING:
            return RuntimeAction(
                "SearchTurn",
                90.0 * self.turn_sign,
                ActionSource.RUNTIME_RECOVERY,
            )
        elif self.stage == RecoveryStage.EVADE_PENDING:
            return RuntimeAction(
                "EvadeBack",
                0.0,
                ActionSource.RUNTIME_RECOVERY,
            )

        return None


    def on_action_sent(self, action: RuntimeAction) -> None:
        if (
            self.stage == RecoveryStage.TURN_PENDING
            and action.name == "SearchTurn"
            and action.source == ActionSource.RUNTIME_RECOVERY
        ):
            self.stage = RecoveryStage.TURN_EXECUTING
            return

        if (
            self.stage == RecoveryStage.EVADE_PENDING
            and action.name == "EvadeBack"
            and action.source == ActionSource.RUNTIME_RECOVERY
        ):
            self.stage = RecoveryStage.EVADE_EXECUTING
            return

        if (
            self.stage == RecoveryStage.IDLE
            and self.verify_pending
            and action.source == ActionSource.POLICY
            and self.verify_action is None
        ):
            self.verify_action = action.name

    def on_action_finished(self, action: RuntimeAction) -> None:
        if (
            self.stage == RecoveryStage.TURN_EXECUTING
            and action.name == "SearchTurn"
            and action.source == ActionSource.RUNTIME_RECOVERY
        ):
            if self.turn_failed:
                self.stage = RecoveryStage.EVADE_PENDING
            else:
                self.stage = RecoveryStage.IDLE
                self.verify_pending = True
                self.verify_action = None
            self.turn_failed = False
            return

        if (
            self.stage == RecoveryStage.EVADE_EXECUTING
            and action.name == "EvadeBack"
            and action.source == ActionSource.RUNTIME_RECOVERY
        ):
            self.stage = RecoveryStage.IDLE
            self.turn_failed = False
            self.verify_pending = False
            self.verify_action = None
            return

        if (
            self.stage == RecoveryStage.IDLE
            and self.verify_pending
            and action.name == self.verify_action
            and action.source == ActionSource.POLICY
            and self.verify_action is not None
        ):
            self.verify_pending = False
            self.verify_action = None
            self.turn_sign *= -1


    def reset(self) -> None:
        self.stage = RecoveryStage.IDLE
        self.turn_sign = 1
        self.turn_failed = False
        self.verify_pending = False
        self.verify_action = None