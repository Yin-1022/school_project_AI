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


# =====================================
# Step 4-2B-2 Integration Tests
# =====================================

def _finish_initial_recovery_turn(controller: RuntimeController) -> None:
    """進入 Recovery，完成 SearchTurn，進入 Policy Verification。"""
    controller.on_cantmove()

    turn = controller.next_action()
    assert turn is not None
    assert turn.name == "SearchTurn"
    assert turn.source == ActionSource.RUNTIME_RECOVERY

    controller.on_action_sent(turn)
    controller.on_action_started()

    result = controller.on_action_ended()
    assert result is not None
    assert result.reason == ExecutionEndReason.COMPLETED
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending is True


@pytest.mark.parametrize(
    "name,value",
    [
        ("Hold", 0.0),
        ("Retreat", 0.0),
        ("EvadeBack", 0.0),
        ("SearchTurn", -50.0),
        ("PatrolStep", 50.0),
    ],
)
def test_nonmovement_policy_preserves_recovery_verification(name, value):
    controller = RuntimeController()
    _finish_initial_recovery_turn(controller)

    policy = RuntimeAction(name, value, ActionSource.POLICY)

    assert controller.next_action(policy) == policy

    controller.on_action_sent(policy)

    # 非驗證用 Movement，不應被記成 verify_action
    assert controller.recovery.verify_action is None

    controller.on_action_started()
    controller.on_action_ended()

    assert controller.recovery.verify_pending is True
    assert controller.recovery.verify_action is None
    assert controller.recovery.turn_sign == 1

    # 真正的 Movement 完成後，才結束 Verification
    movement = RuntimeAction("Advance", 0.0, ActionSource.POLICY)

    assert controller.next_action(movement) == movement
    controller.on_action_sent(movement)
    assert controller.recovery.verify_action == "Advance"

    controller.on_action_started()
    controller.on_action_ended()

    assert controller.recovery.verify_pending is False
    assert controller.recovery.verify_action is None
    assert controller.recovery.turn_sign == -1


def test_rejected_verification_strafe_clears_without_turn_flip():
    controller = RuntimeController()
    _finish_initial_recovery_turn(controller)

    strafe = RuntimeAction("StrafeLeft", 0.0, ActionSource.POLICY)

    assert controller.next_action(strafe) == strafe
    controller.on_action_sent(strafe)
    controller.on_action_started()

    assert controller.recovery.verify_action == "StrafeLeft"

    result = controller.on_notseeplayer()

    assert result is not None
    assert result.reason == ExecutionEndReason.REJECTED
    assert result.action == strafe

    # Rejected 不是成功完成 Verification
    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending is False
    assert controller.recovery.verify_action is None
    assert controller.recovery.turn_sign == 1

    # Rejection 仍然必須排入 Fallback
    assert controller.fallback.pending is True

    fallback = controller.next_action()
    assert fallback is not None
    assert fallback.name == "Advance"
    assert fallback.source == ActionSource.RUNTIME_FALLBACK

    controller.on_action_sent(fallback)
    controller.on_action_started()
    controller.on_action_ended()

    assert controller.fallback.pending is False
    assert controller.recovery.verify_pending is False
    assert controller.recovery.turn_sign == 1


def test_pending_fallback_waits_for_attack_guard():
    controller = RuntimeController()

    strafe = RuntimeAction("StrafeRight", 0.0, ActionSource.POLICY)

    controller.on_action_sent(strafe)
    controller.on_action_started()

    result = controller.on_notseeplayer()
    assert result is not None
    assert result.reason == ExecutionEndReason.REJECTED
    assert controller.fallback.pending is True

    # Fallback Pending 時剛好進入 Attack Lock
    controller.on_attack_start("att2")

    assert controller.can_send() is False
    assert controller.next_action() is None

    # Attack Lock 不應吃掉 Pending Fallback
    assert controller.fallback.pending is True
    assert controller.execution.state.current_action is None

    controller.on_attack_end("att2")

    fallback = controller.next_action()
    assert fallback is not None
    assert fallback.name == "Advance"
    assert fallback.source == ActionSource.RUNTIME_FALLBACK

    controller.on_action_sent(fallback)

    assert controller.fallback.pending is False
    assert controller.execution.state.current_action == fallback

    controller.on_action_started()
    controller.on_action_ended()

    assert controller.can_send() is True


def test_fallback_has_priority_over_pending_hit_reaction():
    controller = RuntimeController()

    strafe = RuntimeAction("StrafeLeft", 0.0, ActionSource.POLICY)

    controller.on_action_sent(strafe)
    controller.on_action_started()

    # Boss 受傷且看不到玩家，排入 Hit Reaction
    controller.on_boss_hit(visible=False)
    assert controller.hit_reaction.pending is True

    # Strafe 同時被拒絕，排入 Fallback
    result = controller.on_notseeplayer()
    assert result is not None
    assert result.reason == ExecutionEndReason.REJECTED
    assert controller.fallback.pending is True

    # Fallback 的優先級高於 Hit Reaction
    first = controller.next_action()

    assert first is not None
    assert first.name == "Advance"
    assert first.source == ActionSource.RUNTIME_FALLBACK

    controller.on_action_sent(first)
    controller.on_action_started()
    controller.on_action_ended()

    assert controller.fallback.pending is False
    assert controller.hit_reaction.pending is True

    # Fallback 結束後才處理 Hit Reaction
    second = controller.next_action()

    assert second is not None
    assert second.name == "SearchTurn"
    assert second.value == 135.0
    assert second.source == ActionSource.RUNTIME_HIT_REACTION

    controller.on_action_sent(second)
    controller.on_action_started()
    controller.on_action_ended()

    assert controller.hit_reaction.pending is False
    assert controller.hit_reaction.attempts == 1


def test_hit_reaction_attempt_limit_and_visibility_reset():
    controller = RuntimeController()

    # 連續兩次受傷，允許兩次 Hit Reaction
    for attempt in (1, 2):
        controller.on_boss_hit(visible=False)

        action = controller.next_action()

        assert action is not None
        assert action.name == "SearchTurn"
        assert action.value == 135.0
        assert action.source == ActionSource.RUNTIME_HIT_REACTION

        controller.on_action_sent(action)
        assert controller.hit_reaction.attempts == attempt

        controller.on_action_started()
        controller.on_action_ended()

    # 第三次受傷，不應再產生 Hit Reaction
    controller.on_boss_hit(visible=False)

    assert controller.next_action() is None
    assert controller.hit_reaction.pending is False
    assert controller.hit_reaction.attempts == 2

    # 重新看到玩家後，重置嘗試次數
    controller.on_visibility_updated(visible=True)

    assert controller.hit_reaction.attempts == 0
    assert controller.hit_reaction.pending is False

    controller.on_boss_hit(visible=False)

    action = controller.next_action()
    assert action is not None
    assert action.source == ActionSource.RUNTIME_HIT_REACTION


@pytest.mark.parametrize(
    "pending_type",
    ["recovery", "fallback", "hit_reaction"],
)
def test_reset_discards_pending_runtime_actions(pending_type):
    controller = RuntimeController()

    if pending_type == "recovery":
        controller.on_cantmove()
        assert controller.recovery.stage == RecoveryStage.TURN_PENDING

    elif pending_type == "fallback":
        strafe = RuntimeAction("StrafeLeft", 0.0, ActionSource.POLICY)
        controller.on_action_sent(strafe)
        controller.on_action_started()

        result = controller.on_notseeplayer()
        assert result is not None
        assert controller.fallback.pending is True

    elif pending_type == "hit_reaction":
        controller.on_boss_hit(visible=False)
        assert controller.hit_reaction.pending is True

    controller.reset()

    assert controller.execution.can_send() is True
    assert controller.can_send() is True
    assert controller.next_action() is None

    assert controller.recovery.stage == RecoveryStage.IDLE
    assert controller.recovery.verify_pending is False
    assert controller.fallback.pending is False
    assert controller.hit_reaction.pending is False
    assert controller.hit_reaction.attempts == 0

    # Reset 後仍可以正常執行 Policy
    policy = RuntimeAction("Advance", 0.0, ActionSource.POLICY)
    assert controller.next_action(policy) == policy

    controller.on_action_sent(policy)
    controller.on_action_started()

    result = controller.on_action_ended()
    assert result is not None
    assert result.reason == ExecutionEndReason.COMPLETED
