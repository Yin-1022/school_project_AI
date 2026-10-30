import pytest

from runtime.stuck_recovery import RecoveryStage, StuckRecoveryController
from runtime.types import ExecutionEndReason, ExecutionResult, RuntimeAction, ActionSource


def test_next_action_does_not_change_stage():
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    action1 = controller.next_action()
    action2 = controller.next_action()

    assert action1 == action2
    assert controller.stage == RecoveryStage.TURN_PENDING


def test_unrelated_action_cannot_finish_recovery():
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    turn = controller.next_action()
    controller.on_action_sent(turn)

    unrelated = RuntimeAction(
        "SearchTurn",
        135.0,
        ActionSource.RUNTIME_HIT_REACTION,
    )

    controller.on_action_finished(unrelated)

    assert controller.stage == RecoveryStage.TURN_EXECUTING


def test_turnPending_after_cantmove():
    controller = StuckRecoveryController()

    # 1. Initial
    assert controller.stage == RecoveryStage.IDLE

    # 2. CantMove received
    controller.on_cantmove(None)

    assert controller.stage == RecoveryStage.TURN_PENDING
    assert not controller.turn_failed


def test_sending_and_confirming_SearchTurn():
    controller = StuckRecoveryController()

    # 1. CantMove received
    controller.on_cantmove(None)

    # 2. Next action should be SearchTurn
    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.source == ActionSource.RUNTIME_RECOVERY

    # 3. Action sent
    controller.on_action_sent(action)

    assert controller.stage == RecoveryStage.TURN_EXECUTING

    # 4. Action finished
    controller.on_action_finished(action)

    assert controller.stage == RecoveryStage.IDLE
    assert controller.verify_pending
    assert controller.verify_action is None


def test_searchTurn_success_and_verify():
    controller = StuckRecoveryController()

    # 1. CantMove received
    controller.on_cantmove(None)

    # 2. Next action should be SearchTurn
    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.source == ActionSource.RUNTIME_RECOVERY

    # 3. Action sent
    controller.on_action_sent(action)

    assert controller.stage == RecoveryStage.TURN_EXECUTING

    # 4. Action finished (success)
    controller.on_action_finished(action)

    assert controller.stage == RecoveryStage.IDLE
    assert controller.verify_pending


def test_searchTurn_cantmove():
    controller = StuckRecoveryController()

    # 1. CantMove received
    controller.on_cantmove(None)

    # 2. Next action should be SearchTurn
    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.source == ActionSource.RUNTIME_RECOVERY

    # 3. Action sent
    controller.on_action_sent(action)
    controller.on_cantmove(action)
    assert controller.turn_failed

    # 4. Action finished (failed)
    controller.on_action_finished(action)
    assert controller.stage == RecoveryStage.EVADE_PENDING

    # 5. Next action sent
    evade_action = controller.next_action()
    assert evade_action is not None
    assert evade_action.name == "EvadeBack"

    controller.on_action_sent(evade_action)
    assert controller.stage == RecoveryStage.EVADE_EXECUTING

    controller.on_action_finished(evade_action)
    assert controller.stage == RecoveryStage.IDLE

def test_movement_once_again_cantmove_after_searchTurn():
    controller = StuckRecoveryController()

    # 1. CantMove received
    controller.on_cantmove(None)

    # 2. Next action should be SearchTurn
    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.source == ActionSource.RUNTIME_RECOVERY

    # 3. Action sent
    controller.on_action_sent(action)

    assert controller.stage == RecoveryStage.TURN_EXECUTING

    # 4. Action finished (success)
    controller.on_action_finished(action)

    assert controller.stage == RecoveryStage.IDLE
    assert controller.verify_pending

    policy_action = RuntimeAction(
        name="Advance",
        value=0.0,
        source=ActionSource.POLICY,
    )

    controller.on_action_sent(policy_action)
    assert controller.verify_action == "Advance"

    # 5. CantMove received again
    controller.on_cantmove(policy_action)

    assert controller.stage == RecoveryStage.EVADE_PENDING
    assert controller.verify_pending is False


def test_flipping_turn_sign():
    controller = StuckRecoveryController()

    # 1. CantMove received
    controller.on_cantmove(None)

    # 2. Next action should be SearchTurn
    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.source == ActionSource.RUNTIME_RECOVERY

    # 3. Action sent
    controller.on_action_sent(action)

    assert controller.stage == RecoveryStage.TURN_EXECUTING

    # 4. Action finished (success)
    controller.on_action_finished(action)

    assert controller.stage == RecoveryStage.IDLE
    assert controller.verify_pending

    # 5. Send a policy action to verify turn sign flipping
    policy_action = RuntimeAction(
        name="Advance",
        value=0.0,
        source=ActionSource.POLICY,
    )
    controller.on_action_sent(policy_action)
    controller.on_action_finished(policy_action)

    assert controller.turn_sign == -1


def test_runtime_action_not_used_for_verification():
    controller = StuckRecoveryController()

    controller.verify_pending = True
    runtime_action = RuntimeAction(
        name="SearchTurn",
        value=90.0,
        source=ActionSource.RUNTIME_HIT_REACTION,
    )

    controller.on_action_sent(runtime_action)

    assert controller.verify_pending
    assert controller.verify_action is None

    controller.on_action_finished(runtime_action)

    assert controller.verify_pending
    assert controller.verify_action is None


def test_policy_action_used_for_verification():
    controller = StuckRecoveryController()
    controller.verify_pending = True

    policy_action = RuntimeAction(
        name="Advance",
        value=0.0,
        source=ActionSource.POLICY,
    )

    controller.on_action_sent(policy_action)

    assert controller.verify_action == "Advance"

    controller.on_action_finished(policy_action)

    assert controller.verify_pending is False
    assert controller.verify_action is None


@pytest.mark.parametrize(
    "reason",
    [
        ExecutionEndReason.START_TIMEOUT,
        ExecutionEndReason.END_TIMEOUT,
    ],
)
def test_search_turn_timeout_enters_evade_pending(reason):
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)
    assert controller.stage == RecoveryStage.TURN_EXECUTING

    result = ExecutionResult(action=turn, reason=reason)
    controller.on_action_failed(result)

    assert controller.stage == RecoveryStage.EVADE_PENDING
    assert controller.turn_failed is False
    assert controller.verify_pending is False
    assert controller.verify_action is None

    evade = controller.next_action()
    assert evade is not None
    assert evade.name == "EvadeBack"
    assert evade.source == ActionSource.RUNTIME_RECOVERY


@pytest.mark.parametrize(
    "reason",
    [
        ExecutionEndReason.START_TIMEOUT,
        ExecutionEndReason.END_TIMEOUT,
    ],
)
def test_evade_back_timeout_ends_recovery(reason):
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)
    controller.on_cantmove(turn)
    controller.on_action_finished(turn)

    assert controller.stage == RecoveryStage.EVADE_PENDING

    evade = controller.next_action()
    assert evade is not None
    controller.on_action_sent(evade)
    assert controller.stage == RecoveryStage.EVADE_EXECUTING

    controller.on_action_failed(ExecutionResult(evade, reason))

    assert controller.stage == RecoveryStage.IDLE
    assert controller.next_action() is None
    assert controller.verify_pending is False
    assert controller.turn_failed is False


@pytest.mark.parametrize(
    "reason",
    [
        ExecutionEndReason.START_TIMEOUT,
        ExecutionEndReason.END_TIMEOUT,
    ],
)
def test_policy_verification_timeout_clears_verification(reason):
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)
    controller.on_action_finished(turn)

    assert controller.stage == RecoveryStage.IDLE
    assert controller.verify_pending

    policy = RuntimeAction(
        "Advance", 0.0, ActionSource.POLICY
    )
    controller.on_action_sent(policy)
    assert controller.verify_action == "Advance"

    original_sign = controller.turn_sign
    controller.on_action_failed(ExecutionResult(policy, reason))

    assert controller.stage == RecoveryStage.IDLE
    assert controller.verify_pending is False
    assert controller.verify_action is None
    assert controller.turn_sign == original_sign


def test_unrelated_timeout_does_not_change_recovery():
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)

    unrelated = RuntimeAction(
        "Advance", 0.0, ActionSource.POLICY
    )
    controller.on_action_failed(
        ExecutionResult(
            unrelated,
            ExecutionEndReason.START_TIMEOUT,
        )
    )

    assert controller.stage == RecoveryStage.TURN_EXECUTING
    assert controller.turn_failed is False


def test_non_timeout_failure_does_not_change_recovery():
    controller = StuckRecoveryController()
    controller.on_cantmove(None)

    turn = controller.next_action()
    assert turn is not None
    controller.on_action_sent(turn)

    controller.on_action_failed(
        ExecutionResult(
            turn,
            ExecutionEndReason.REJECTED,
        )
    )

    assert controller.stage == RecoveryStage.TURN_EXECUTING



def test_reset():
    controller = StuckRecoveryController()

    # 1. CantMove received
    controller.on_cantmove(None)

    # 2. Next action should be SearchTurn
    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.source == ActionSource.RUNTIME_RECOVERY

    # 3. Action sent
    controller.on_action_sent(action)

    assert controller.stage == RecoveryStage.TURN_EXECUTING

    # 4. Reset the controller
    controller.reset()

    assert controller.stage == RecoveryStage.IDLE
    assert not controller.verify_pending
    assert controller.verify_action is None
    assert controller.turn_sign == 1
    assert not controller.turn_failed
