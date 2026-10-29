from runtime.controller import RuntimeController
from runtime.types import RuntimeAction, ActionSource


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