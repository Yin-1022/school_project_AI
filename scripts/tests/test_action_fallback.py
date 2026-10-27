from runtime.action_fallback import ActionFallbackController
from runtime.types import RuntimeAction, ActionSource

def test_pending_when_strafe_and_notseeplayer():
    controller = ActionFallbackController()

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.POLICY,
    )

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=current_action,
    )
    assert on_notseeplayer_result is True
    assert controller.pending is True

    controller.reset()

    current_action = RuntimeAction(
        name="StrafeRight",
        value=0.0,
        source=ActionSource.POLICY,
    )

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=current_action,
    )
    assert on_notseeplayer_result is True
    assert controller.pending is True


def test_pending_when_advance_and_notseeplayer():
    controller = ActionFallbackController()

    current_action = RuntimeAction(
        name="Advance",
        value=0.0,
        source=ActionSource.POLICY,
    )

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=current_action,
    )
    assert on_notseeplayer_result is False
    assert controller.pending is False


def test_pending_when_none_and_notseeplayer():
    controller = ActionFallbackController()

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=None,
    )
    assert on_notseeplayer_result is False
    assert controller.pending is False


def test_pending_when_recovery_and_notseeplayer():
    controller = ActionFallbackController()

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.RUNTIME_RECOVERY,
    )

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=current_action,
    )
    assert on_notseeplayer_result is False
    assert controller.pending is False


def test_no_repeat_fallback_when_repeat_notseeplayer():
    controller = ActionFallbackController()

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.POLICY,
    )

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=current_action,
    )
    assert on_notseeplayer_result is True
    assert controller.pending is True

    on_notseeplayer_result = controller.on_notseeplayer(
        current_action=current_action,
    )
    assert on_notseeplayer_result is False
    assert controller.pending is True


def test_next_action_not_changing_pending():
    controller = ActionFallbackController(fallback_value=1.0)

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.POLICY,
    )

    controller.on_notseeplayer(current_action=current_action)
    next_action = controller.next_action()
    assert next_action is not None
    assert next_action.name == "Advance"
    assert next_action.value == 1.0
    assert next_action.source == ActionSource.RUNTIME_FALLBACK
    assert controller.pending is True


def test_clean_after_sending_fallback_action():
    controller = ActionFallbackController(fallback_value=1.0)

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.POLICY,
    )

    controller.on_notseeplayer(current_action=current_action)
    next_action = controller.next_action()
    controller.on_action_sent(next_action)
    assert controller.pending is False

def test_no_clean_after_sending_non_fallback_action():
    controller = ActionFallbackController(fallback_value=1.0)

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.POLICY,
    )

    controller.on_notseeplayer(current_action=current_action)
    next_action = controller.next_action()

    different_action = RuntimeAction(
        name="Advance",
        value=2.0,
        source=ActionSource.POLICY,
    )
    controller.on_action_sent(different_action)
    assert controller.pending is True


def test_reset():
    controller = ActionFallbackController(fallback_value=1.0)

    current_action = RuntimeAction(
        name="StrafeLeft",
        value=0.0,
        source=ActionSource.POLICY,
    )

    controller.on_notseeplayer(current_action=current_action)
    assert controller.pending is True

    controller.reset()
    assert controller.pending is False