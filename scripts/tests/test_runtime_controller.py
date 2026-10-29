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