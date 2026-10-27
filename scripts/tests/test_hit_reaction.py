from runtime.hit_reaction import HitReactionController
from runtime.types import RuntimeAction, ActionSource


def test_pending_when_hit_and_player_not_visible():
    controller = HitReactionController()

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=False)
    assert controller.pending


def test_hit_and_player_visible():
    controller = HitReactionController()

    controller.on_visible_update(visible=True)
    controller.on_boss_hit(visible=True, recovery_active=False)
    assert not controller.pending


def test_hit_when_recovery_active():
    controller = HitReactionController()

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=True)
    assert not controller.pending


def test_next_action():
    controller = HitReactionController(turn_angle=135.0)

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=False)

    action = controller.next_action()
    assert action is not None
    assert action.name == "SearchTurn"
    assert action.value == 135.0
    assert action.source == ActionSource.RUNTIME_RECOVERY
    assert controller.pending


def test_on_attempt_adding_after_hit_reaction():
    controller = HitReactionController(turn_angle=135.0)

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=False)

    action = controller.next_action()

    controller.on_action_sent(action)
    assert controller.attempts == 1


def test_max_two_attempts():
    controller = HitReactionController(max_attempts=2)

    for _ in range(2):
        controller.on_boss_hit(
            visible=False,
            recovery_active=False,
        )

        action = controller.next_action()
        assert action is not None

        controller.on_action_sent(action)

    assert controller.attempts == 2

    controller.on_boss_hit(
        visible=False,
        recovery_active=False,
    )

    assert controller.next_action() is None


def test_reseeing_player_resets_attempts():
    controller = HitReactionController(turn_angle=135.0, max_attempts=2)

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=False)

    action = controller.next_action()
    controller.on_action_sent(action)
    assert controller.attempts == 1

    # Re-seeing player resets attempts
    controller.on_visible_update(visible=True)
    assert controller.attempts == 0
    assert not controller.pending


def test_SearchTurn_from_other_sources():
    controller = HitReactionController()

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=False)

    action = controller.next_action()
    assert action is not None

    # Simulate sending an action from a different source
    other_action = RuntimeAction(
        name="SearchTurn",
        value=90.0,
        source=ActionSource.POLICY,
    )
    controller.on_action_sent(other_action)

    # The pending state should remain unchanged
    assert controller.pending
    assert controller.attempts == 0


def test_reset():
    controller = HitReactionController(turn_angle=135.0, max_attempts=2)

    controller.on_visible_update(visible=False)
    controller.on_boss_hit(visible=False, recovery_active=False)

    action = controller.next_action()
    controller.on_action_sent(action)
    assert controller.attempts == 1

    # Reset the controller
    controller.reset()
    assert controller.attempts == 0
    assert not controller.pending