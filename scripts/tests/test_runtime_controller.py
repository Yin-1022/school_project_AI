import pytest
from runtime.controller import RuntimeController
from runtime.types import ExecutionEndReason, RuntimeAction, ActionSource
from runtime.stuck_recovery import RecoveryStage


def test_recovery_has_highest_priority():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    controller.hit_reaction.on_boss_hit(
        visible=False,
        recovery_active=False,
    )

    controller.fallback.on_notseeplayer(
        RuntimeAction(
            "StrafeLeft",
            0.0,
            ActionSource.POLICY,
        )
    )

    controller.recovery.on_cantmove(None)

    action = controller.next_action(policy)

    assert action is not None
    assert action.name == "SearchTurn"
    assert action.value == 90.0
    assert action.source == ActionSource.RUNTIME_RECOVERY


def test_correctly_policy_action_return_in_init():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    action = controller.next_action(policy)

    assert action is not None
    assert action.name == "Advance"
    assert action.value == 0.0
    assert action.source == ActionSource.POLICY


def test_no_action_when_execution_busy():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    controller.on_action_sent(policy)

    action = controller.next_action(policy)

    assert action is None


def test_stop_action_when_attack_active():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    controller.on_attack_start("att1")

    assert not controller.can_send()
    assert controller.next_action(policy) is None

    controller.on_attack_end("att1")

    assert controller.can_send()
    assert controller.next_action(policy) == policy


def test_fallback_prior_than_hit_reaction():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    controller.fallback.on_notseeplayer(
        RuntimeAction(
            "StrafeLeft",
            0.0,
            ActionSource.POLICY,
        )
    )

    controller.hit_reaction.on_boss_hit(
        visible=False,
        recovery_active=False,
    )

    action = controller.next_action(policy)

    assert action is not None
    assert action.name == "Advance"
    assert action.source == ActionSource.RUNTIME_FALLBACK


def test_hit_reaction_prior_than_policy():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    controller.hit_reaction.on_boss_hit(
        visible=False,
        recovery_active=False,
    )

    action = controller.next_action(policy)

    assert action is not None
    assert action.name == "SearchTurn"
    assert action.value == 135.0
    assert action.source == ActionSource.RUNTIME_HIT_REACTION


def test_actend_unlock_and_notify_recovery():
    controller = RuntimeController()

    controller.on_cantmove()

    assert controller.recovery.stage == RecoveryStage.TURN_PENDING

    action = controller.next_action()
    assert action is not None

    controller.on_action_sent(action)
    assert controller.recovery.stage == RecoveryStage.TURN_EXECUTING

    controller.on_action_started()

    result = controller.on_action_ended()

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.COMPLETED

    assert controller.can_send()
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending


def test_rejected_when_notseeplayer():
    controller = RuntimeController()

    action = RuntimeAction(
        "StrafeLeft",
        0.0,
        ActionSource.POLICY,
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    result = controller.on_notseeplayer()

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.REJECTED


def test_fallback_advance_after_rejected_strafe():
    controller = RuntimeController()

    action = RuntimeAction(
        "StrafeLeft",
        0.0,
        ActionSource.POLICY,
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    result = controller.on_notseeplayer()

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.REJECTED

    next_action = controller.next_action()

    assert next_action is not None
    assert next_action.name == "Advance"



import pytest
from runtime.stuck_recovery import RecoveryStage


@pytest.mark.parametrize(
    "reason",
    [
        ExecutionEndReason.START_TIMEOUT,
        ExecutionEndReason.END_TIMEOUT,
    ],
)
def test_tick_routes_search_turn_timeout_to_recovery(reason):
    controller = RuntimeController()
    controller.on_cantmove()

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)

    if reason == ExecutionEndReason.END_TIMEOUT:
        controller.on_action_started()
        started_at = controller.execution.state.started_at
        assert started_at is not None
        now = started_at + controller.execution.end_timeout
    else:
        sent_at = controller.execution.state.sent_at
        assert sent_at is not None
        now = sent_at + controller.execution.start_timeout

    result = controller.tick(now=now)

    assert result is not None
    assert result.action == turn
    assert result.reason == reason

    assert controller.execution.can_send()
    assert controller.recovery.stage == RecoveryStage.EVADE_PENDING

    next_action = controller.next_action()
    assert next_action is not None
    assert next_action.name == "EvadeBack"
    assert next_action.source == ActionSource.RUNTIME_RECOVERY


def test_tick_routes_evade_timeout_to_recovery():
    controller = RuntimeController()
    controller.on_cantmove()

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)
    controller.on_cantmove()
    controller.on_action_ended()

    assert controller.recovery.stage == RecoveryStage.EVADE_PENDING

    evade = controller.next_action()
    assert evade is not None
    controller.on_action_sent(evade)

    sent_at = controller.execution.state.sent_at
    assert sent_at is not None

    result = controller.tick(
        now=sent_at + controller.execution.start_timeout
    )

    assert result is not None
    assert result.reason == ExecutionEndReason.START_TIMEOUT

    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.execution.can_send()


def test_tick_policy_verification_timeout():
    controller = RuntimeController()
    controller.on_cantmove()

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)
    controller.on_action_ended()

    assert controller.recovery.verify_pending

    policy = RuntimeAction(
        "Advance", 0.0, ActionSource.POLICY
    )
    controller.on_action_sent(policy)

    sent_at = controller.execution.state.sent_at
    assert sent_at is not None

    original_sign = controller.recovery.turn_sign

    result = controller.tick(
        now=sent_at + controller.execution.start_timeout
    )

    assert result is not None
    assert result.action == policy
    assert result.reason == ExecutionEndReason.START_TIMEOUT
    assert controller.recovery.verify_pending is False
    assert controller.recovery.verify_action is None
    assert controller.recovery.turn_sign == original_sign


def test_recovery_start_cancels_hit_reaction_pending():
    controller = RuntimeController()

    controller.on_boss_hit(visible=False)
    assert controller.hit_reaction.pending

    controller.on_cantmove()

    assert controller.recovery.stage == RecoveryStage.TURN_PENDING
    assert controller.hit_reaction.pending is False
    assert controller.hit_reaction.next_action() is None


def test_recovery_start_cancels_fallback_pending():
    controller = RuntimeController()

    strafe = RuntimeAction(
        "StrafeLeft", 0.0, ActionSource.POLICY
    )

    controller.on_action_sent(strafe)

    rejected = controller.on_notseeplayer()
    assert rejected is not None
    assert rejected.reason == ExecutionEndReason.REJECTED
    assert controller.fallback.pending

    controller.on_cantmove()

    assert controller.recovery.stage == RecoveryStage.TURN_PENDING
    assert controller.fallback.pending is False
    assert controller.fallback.rejected_action is None

    # 原先的拒絕結果仍然保留在呼叫端
    assert rejected.action == strafe
    assert rejected.reason == ExecutionEndReason.REJECTED


def test_recovery_verification_blocks_new_hit_reaction():
    controller = RuntimeController()
    controller.on_cantmove()

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)
    controller.on_action_ended()

    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending

    controller.on_boss_hit(visible=False)

    assert controller.hit_reaction.pending is False
    assert controller.hit_reaction.next_action() is None


def test_tick_clears_expired_attack_lock():
    controller = RuntimeController()

    controller.on_attack_start("att1")
    assert controller.attack.is_attacking()
    assert not controller.can_send()

    deadline = controller.attack.active_deadline["att1"]

    result = controller.tick(now=deadline + 0.01)

    assert result is None
    assert not controller.attack.is_attacking()
    assert controller.can_send()


def test_tick_does_not_clear_attack_early():
    controller = RuntimeController()

    controller.on_attack_start("att2")
    deadline = controller.attack.active_deadline["att2"]

    controller.tick(now=deadline - 0.01)

    assert controller.attack.is_attacking()
    assert not controller.can_send()


def test_tick_without_timeout_returns_none():
    controller = RuntimeController()

    policy = RuntimeAction(
        "Advance", 0.0, ActionSource.POLICY
    )
    controller.on_action_sent(policy)

    sent_at = controller.execution.state.sent_at
    assert sent_at is not None

    result = controller.tick(now=sent_at + 0.1)

    assert result is None
    assert not controller.execution.can_send()


def test_reset_clears_all_active_components():
    controller = RuntimeController()

    controller.on_attack_start("att1")
    controller.on_boss_hit(visible=False)

    strafe = RuntimeAction(
        "StrafeLeft", 0.0, ActionSource.POLICY
    )
    controller.fallback.on_notseeplayer(strafe)

    controller.on_cantmove()

    turn = controller.next_action()

    # Attack Lock 尚在，因此目前不應送出動作
    assert turn is None

    controller.reset()

    assert controller.execution.can_send()
    assert not controller.attack.is_attacking()
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending is False
    assert controller.hit_reaction.pending is False
    assert controller.hit_reaction.attempts == 0
    assert controller.fallback.pending is False
    assert controller.fallback.rejected_action is None


def test_reset_five_controllers():
    controller = RuntimeController()
    controller.on_attack_start("att1")

    controller.reset()

    assert controller.execution.can_send()
    assert not controller.attack.is_attacking()

    assert controller.recovery.stage == RecoveryStage.IDLE
    assert not controller.recovery.verify_pending

    assert not controller.hit_reaction.pending
    assert controller.hit_reaction.attempts == 0

    assert not controller.fallback.pending
    assert controller.fallback.rejected_action is None


def test_deny_non_policy_source_action():
    controller = RuntimeController()

    non_policy_action = RuntimeAction(
        "EvadeBack", 0.0, ActionSource.RUNTIME_RECOVERY
    )

    with pytest.raises(
        ValueError,
        match="policy_action must have POLICY source",
    ):
        controller.next_action(non_policy_action)