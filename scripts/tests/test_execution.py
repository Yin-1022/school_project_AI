import pytest
from runtime.execution import ExecutionController
from runtime.types import ExecutionEndReason, RuntimeAction, ActionSource


def test_execution_lifecycle():
    controller = ExecutionController()

    action = RuntimeAction(
        name="Advance",
        value=0.0,
        source=ActionSource.POLICY,
    )

    # 1. Initial
    assert controller.can_send()
    assert controller.state.current_action is None

    # 2. Action sent
    controller.on_action_sent(action)

    assert not controller.can_send()
    assert controller.state.busy
    assert controller.state.waiting_start
    assert controller.state.current_action == action
    assert controller.state.sent_at is not None

    # 3. Action started
    controller.on_action_started()

    assert controller.state.busy
    assert not controller.state.waiting_start
    assert controller.state.started_at is not None

    # 4. Action ended
    result = controller.on_action_end()

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.COMPLETED
    assert controller.can_send()
    assert controller.state.current_action is None
    assert controller.state.sent_at is None
    assert controller.state.started_at is None


def test_invalid_events():
    controller = ExecutionController()

    # Idle 時收到 End，應忽略
    assert controller.on_action_end() is None

    # Idle 時收到 Start，應忽略
    controller.on_action_started()
    assert controller.can_send()

    action = RuntimeAction(
        "SearchTurn",
        90.0,
        ActionSource.RUNTIME_RECOVERY,
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    first_started_at = controller.state.started_at

    # 重複 Start 不應覆蓋時間
    controller.on_action_started()
    assert controller.state.started_at == first_started_at

    # Busy 時不能再送 Action
    try:
        controller.on_action_sent(action)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass

    # 不應覆蓋正在執行的 Action
    assert controller.state.current_action == action


def test_reset():
    controller = ExecutionController()

    action = RuntimeAction(
        "EvadeBack",
        0.0,
        ActionSource.RUNTIME_RECOVERY,
    )

    controller.on_action_sent(action)
    controller.on_action_started()
    controller.reset()

    assert controller.can_send()
    assert controller.state.current_action is None
    assert controller.state.started_at is None


def test_action_start_timeout():
    controller = ExecutionController(
        start_timeout=1.0,
        end_timeout=5.0,
    )

    action = RuntimeAction(
        "SearchTurn", 90.0, ActionSource.RUNTIME_RECOVERY
    )

    controller.on_action_sent(action)
    sent_at = controller.state.sent_at
    assert sent_at is not None

    # 尚未超時
    assert controller.check_timeout(now=sent_at + 0.5) is None
    assert not controller.can_send()

    # 超過 Start Timeout
    result = controller.check_timeout(now=sent_at + 1.0)

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.START_TIMEOUT
    assert controller.can_send()


def test_action_end_timeout():
    controller = ExecutionController(
        start_timeout=1.0,
        end_timeout=5.0,
    )

    action = RuntimeAction(
        "SearchTurn", 90.0, ActionSource.RUNTIME_RECOVERY
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    started_at = controller.state.started_at
    assert started_at is not None

    # 尚未超時
    assert controller.check_timeout(now=started_at + 4.0) is None
    assert not controller.can_send()

    # 超過 End Timeout
    result = controller.check_timeout(now=started_at + 5.0)

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.END_TIMEOUT
    assert controller.can_send()


def test_no_timeout_when_idle():
    controller = ExecutionController()

    assert controller.check_timeout(now=100.0) is None
    assert controller.can_send()


def test_abort_with_rejected_action():
    controller = ExecutionController()

    action = RuntimeAction(
        "EvadeBack", 0.0, ActionSource.RUNTIME_RECOVERY
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    result = controller.abort(ExecutionEndReason.REJECTED)

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.REJECTED
    assert controller.can_send()


def test_abort_while_idle():
    controller = ExecutionController()

    result = controller.abort(ExecutionEndReason.REJECTED)

    assert result is None
    assert controller.can_send()


def test_actend_after_rejected():
    controller = ExecutionController()

    action = RuntimeAction(
        "EvadeBack", 0.0, ActionSource.RUNTIME_RECOVERY
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    result = controller.abort(ExecutionEndReason.REJECTED)

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.REJECTED
    assert controller.can_send()

    # After abort, on_action_end should return None
    assert controller.on_action_end() is None


def test_rejected_action_not_completed_twice():
    controller = ExecutionController()

    action = RuntimeAction(
        "StrafeLeft",
        0.0,
        ActionSource.POLICY,
    )

    controller.on_action_sent(action)

    result = controller.abort(
        ExecutionEndReason.REJECTED
    )

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.REJECTED

    assert controller.on_action_end() is None
    assert controller.can_send()


def test_check_again_after_timeout():
    controller = ExecutionController(
        start_timeout=1.0,
        end_timeout=5.0,
    )

    action=RuntimeAction(
        "SearchTurn",
        90.0,
        ActionSource.RUNTIME_RECOVERY,
    )

    controller.on_action_sent(action)
    sent_at = controller.state.sent_at
    assert sent_at is not None

    result = controller.check_timeout(now=sent_at + 1.0)

    assert result is not None
    assert result.action == action
    assert result.reason == ExecutionEndReason.START_TIMEOUT
    assert controller.can_send()

    assert controller.check_timeout(now=sent_at + 2.0) is None


def test_invalid_abort_reason():
    controller = ExecutionController()

    action = RuntimeAction(
        "EvadeBack", 0.0, ActionSource.RUNTIME_RECOVERY
    )

    controller.on_action_sent(action)
    controller.on_action_started()

    with pytest.raises(ValueError):
        controller.abort(ExecutionEndReason.COMPLETED)


def test_reset():
    controller = ExecutionController()

    action = RuntimeAction(
        "EvadeBack", 0.0, ActionSource.RUNTIME_RECOVERY
    )

    controller.on_action_sent(action)
    controller.on_action_started()
    controller.reset()

    assert controller.can_send()
    assert controller.state.current_action is None
    assert controller.state.started_at is None