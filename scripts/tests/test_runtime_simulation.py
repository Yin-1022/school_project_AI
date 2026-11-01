import pytest
from runtime.controller import RuntimeController
from runtime.types import ExecutionEndReason, ExecutionResult, RuntimeAction, ActionSource
from runtime.stuck_recovery import RecoveryStage

def test_normal_policy_lifecycle():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    action = controller.next_action(policy)
    assert action is not None

    controller.on_action_sent(action)
    assert controller.execution.state.current_action == policy
    assert controller.can_send() is False
    assert controller.next_action() is None

    controller.on_action_started()
    result = controller.on_action_ended()
    assert result is not None
    assert result.reason == ExecutionEndReason.COMPLETED
    assert result.action.source == ActionSource.POLICY

    policy = controller.next_action(policy)
    assert policy is not None
    controller.on_action_sent(policy)
    controller.on_action_started()
    controller.on_action_ended()

    assert controller.can_send() is True
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.fallback.pending is False
    assert controller.hit_reaction.pending is False


def test_stuck_recovery_success_lifecycle():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )
    
    controller.on_cantmove()
    action = controller.next_action(policy)

    assert action is not None
    assert action.name == "SearchTurn"
    assert action.value == 90.0
    assert action.source == ActionSource.RUNTIME_RECOVERY
    assert controller.recovery.stage == RecoveryStage.TURN_PENDING

    controller.on_action_sent(action)
    controller.on_action_started()
    assert controller.recovery.stage == RecoveryStage.TURN_EXECUTING
    controller.on_action_ended()
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending
    assert controller.recovery.verify_action is None

    policy = controller.next_action(policy)
    assert policy is not None
    controller.on_action_sent(policy)
    controller.on_action_started()
    controller.on_action_ended()

    assert controller.recovery.verify_pending is False
    assert controller.recovery.turn_sign == -1

    policy = controller.next_action(policy)
    assert policy is not None
    controller.on_action_sent(policy)
    controller.on_action_started()
    result = controller.on_action_ended()
    assert result is not None
    assert result.reason == ExecutionEndReason.COMPLETED


def test_stuck_recovery_failed_verification():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )
    
    controller.on_cantmove()
    action = controller.next_action(policy)

    assert action is not None
    assert action.name == "SearchTurn"
    assert action.value == 90.0
    assert action.source == ActionSource.RUNTIME_RECOVERY
    assert controller.recovery.stage == RecoveryStage.TURN_PENDING

    controller.on_action_sent(action)
    controller.on_action_started()
    controller.on_action_ended()
    assert controller.recovery.stage == RecoveryStage.IDLE

    policy_action = controller.next_action(policy)
    assert policy_action is not None
    controller.on_action_sent(policy_action)
    assert controller.recovery.verify_action == "Advance"
    controller.on_action_started()
    controller.on_cantmove()
    assert controller.next_action(policy) is None
    assert controller.recovery.stage == RecoveryStage.EVADE_PENDING
    controller.on_action_ended()

    evade_action = controller.next_action(policy)
    assert evade_action.name == "EvadeBack"

    controller.on_action_sent(evade_action)
    assert controller.recovery.stage == RecoveryStage.EVADE_EXECUTING
    controller.on_action_started()

    controller.on_action_ended()
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending == False
    assert controller.recovery.turn_sign == 1


def test_strafe_rejection_fallback_lifecycle():
    controller = RuntimeController()

    strafe = RuntimeAction(
        "StrafeLeft",
        0.0,
        ActionSource.POLICY,
    )

    next_action = controller.next_action(strafe)
    assert next_action is not None
    controller.on_action_sent(next_action)
    controller.on_action_started()

    result = controller.on_notseeplayer()
    assert result is not None
    assert result.action.name == "StrafeLeft"
    assert result.reason == ExecutionEndReason.REJECTED
    assert controller.execution.can_send() is True

    next_action = controller.next_action(strafe)
    assert next_action is not None
    assert next_action.name == "Advance"
    assert next_action.value == 0.0
    assert next_action.source == ActionSource.RUNTIME_FALLBACK
    assert controller.fallback.pending is True

    controller.on_action_sent(next_action)
    controller.on_action_started()
    result = controller.on_action_ended()

    assert controller.fallback.pending is False
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert result.action.source == ActionSource.RUNTIME_FALLBACK
    assert result.reason == ExecutionEndReason.COMPLETED
    assert controller.can_send() is True

    next_action = controller.next_action(strafe)
    assert next_action is not None
    assert next_action.name == "StrafeLeft"
    assert next_action.source == ActionSource.POLICY


def test_attack_lock_watchdog_lifecycle():
    controller = RuntimeController()

    controller.on_attack_start("att1")
    assert controller.attack.is_attacking() is True

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    assert controller.can_send() is False
    assert controller.next_action(policy) is None

    deadline = controller.attack.active_deadline["att1"]

    result = controller.tick(now=deadline - 0.01)
    assert controller.attack.is_attacking() is True
    assert controller.can_send() is False
    assert result is None

    result = controller.tick(now=deadline)
    assert result is None
    assert controller.attack.is_attacking() is False
    assert controller.can_send() is True

    policy = controller.next_action(policy)
    controller.on_action_sent(policy)
    assert controller.execution.state.current_action == policy


def test_mixed_events_recover_to_policy():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    controller.on_visibility_updated(visible=False)
    controller.on_boss_hit(visible=False)

    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.value == 135.0
    assert action.source == ActionSource.RUNTIME_HIT_REACTION
    assert controller.hit_reaction.pending

    controller.on_cantmove()
    assert controller.hit_reaction.pending is False
    turn = controller.next_action(policy)
    assert turn is not None
    assert turn.name == "SearchTurn"
    assert turn.source == ActionSource.RUNTIME_RECOVERY
    assert controller.recovery.stage == RecoveryStage.TURN_PENDING

    controller.on_action_sent(turn)
    assert controller.recovery.stage == RecoveryStage.TURN_EXECUTING
    controller.on_action_started()
    started_at = controller.execution.state.started_at
    result = controller.tick(now=started_at + controller.execution.end_timeout)
    assert result is not None
    assert result.action.name == "SearchTurn"
    assert result.reason == ExecutionEndReason.END_TIMEOUT

    assert controller.recovery.stage == RecoveryStage.EVADE_PENDING
    assert controller.recovery.turn_failed is False
    assert controller.recovery.verify_pending is False
    assert controller.recovery.verify_action is None
    assert controller.can_send() is True
    assert controller.hit_reaction.pending is False

    evade = controller.next_action(policy)
    assert evade is not None
    assert evade.name == "EvadeBack"
    assert evade.value == 0.0
    assert evade.source == ActionSource.RUNTIME_RECOVERY
    assert evade != policy

    controller.on_action_sent(evade)
    controller.on_action_started()
    assert controller.recovery.stage == RecoveryStage.EVADE_EXECUTING

    result = controller.on_action_ended()

    assert result is not None
    assert result.reason == ExecutionEndReason.COMPLETED
    assert result.action.name == "EvadeBack"
    assert result.action.source == ActionSource.RUNTIME_RECOVERY

    assert controller.execution.can_send() is True
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.hit_reaction.pending is False
    assert controller.fallback.pending is False

    next_action = controller.next_action(policy)
    assert next_action == policy
    assert next_action.source == ActionSource.POLICY