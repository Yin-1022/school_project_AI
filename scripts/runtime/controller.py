from .execution import ExecutionController
from .attack import AttackGuard
from .stuck_recovery import RecoveryStage, StuckRecoveryController
from .hit_reaction import HitReactionController
from .action_fallback import ActionFallbackController
from .types import RuntimeAction, ExecutionResult, ExecutionEndReason


class RuntimeController:
    def __init__(self) -> None:
        self.execution = ExecutionController()
        self.attack = AttackGuard()
        self.recovery = StuckRecoveryController()
        self.hit_reaction = HitReactionController()
        self.fallback = ActionFallbackController()

    def can_send(self) -> bool:
        return (
            self.execution.can_send()
            and not self.attack.is_attacking()
        )

    def next_action(
        self,
        policy_action: RuntimeAction | None = None,
    ) -> RuntimeAction | None:
        if not self.can_send():
            return None

        for controller in [
            self.recovery,
            self.fallback,
            self.hit_reaction,
        ]:
            action = controller.next_action()
            if action is not None:
                return action

        return policy_action

    def on_action_sent(self, action: RuntimeAction) -> None:
        self.execution.on_action_sent(action)

        self.recovery.on_action_sent(action)
        self.hit_reaction.on_action_sent(action)
        self.fallback.on_action_sent(action)

    def on_action_started(self) -> None:
        self.execution.on_action_started()

    def on_action_ended(self) -> ExecutionResult | None:
        result = self.execution.on_action_end()

        if result is None:
            return None

        if result.reason == ExecutionEndReason.COMPLETED:
            self.recovery.on_action_finished(result.action)

        return result

    def on_attack_start(self, attack_type: str) -> None:
        self.attack.on_attack_start(attack_type)


    def on_attack_end(self, attack_type: str) -> None:
        self.attack.on_attack_end(attack_type)


    def on_visibility_updated(self, visible: bool) -> None:
        self.hit_reaction.on_visible_update(visible)

    def on_boss_hit(self, visible: bool) -> None:
        self.hit_reaction.on_boss_hit(
            visible=visible,
            recovery_active=(
                self.recovery.stage != RecoveryStage.IDLE
            ),
        )

    def on_cantmove(self) -> None:
        self.recovery.on_cantmove(action=self.execution.state.current_action)

    def on_notseeplayer(self) -> ExecutionResult | None:
        current_action = self.execution.state.current_action

        rejected = self.fallback.on_notseeplayer(current_action=current_action)

        if not rejected:
            return None

        return self.execution.abort(ExecutionEndReason.REJECTED)

    def check_timeouts(
        self,
        now: float | None = None,
    ) -> ExecutionResult | None:
        return self.execution.check_timeout(now=now)

    def reset(self) -> None:
        self.execution.reset()
        self.recovery.reset()
        self.hit_reaction.reset()
        self.fallback.reset()
        self.attack.reset()