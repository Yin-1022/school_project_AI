
from episode_finalizer import finalize_episode
from ground_truth_audit import GroundTruthAudit
from episode_metrics import EpisodeEventAudit, PendingCombatEvents
from runtime.controller import RuntimeController
from runtime.types import ExecutionEndReason
from transition_tracker import TransitionTracker


def handle_rejection_pulse(
    *,
    pulse: bool,
    runtime: RuntimeController,
    tracker: TransitionTracker,
    ground_truth_audit: GroundTruthAudit,
    event_lock,
) -> bool:
    """Reconcile a rejected Strafe immediately."""

    if not pulse:
        return False

    result = runtime.on_notseeplayer()

    if result is None:
        return False

    if result.reason != ExecutionEndReason.REJECTED:
        raise RuntimeError(
            f"Unexpected rejection reason: {result.reason}"
        )

    # The Tracker owns Policy Cache and combat attribution.
    tracker.discard_rejected_policy(
        expected_action=result.action,
    )

    # The Audit owns execution records.
    with event_lock:
        current = ground_truth_audit.current

        if current is not None:
            if current.expected_runtime_action == result.action:
                ground_truth_audit.abort_action(
                    "notseeplayer_rejected"
                )
            else:
                print(
                    "[audit warning] rejected action mismatch: "
                    f"expected={current.expected_runtime_action}, "
                    f"received={result.action}"
                )

    print(
        f"[execution] {result.action.name} rejected "
        "-> runtime fallback pending"
    )

    return True


def finalize_terminal_episode(
    *,
    runtime: RuntimeController,
    tracker: TransitionTracker,
    event_audit: EpisodeEventAudit,
    pending_combat: PendingCombatEvents,
    ground_truth_audit: GroundTruthAudit,
    event_lock,
    event_state: dict,
    episode_result: int,
    policy_step: int,
) -> dict:
    """Reconcile pending rejection before finalizing an episode."""

    if episode_result not in (-1, 1):
        raise ValueError(
            f"Invalid terminal episode result: {episode_result}"
        )

    # Consume any rejection arriving after the last frame poll.
    with event_lock:
        rejection_pulse = bool(
            event_state["notseeplayer_pulse"]
        )
        event_state["notseeplayer_pulse"] = False

    handle_rejection_pulse(
        pulse=rejection_pulse,
        runtime=runtime,
        tracker=tracker,
        ground_truth_audit=ground_truth_audit,
        event_lock=event_lock,
    )

    summary = finalize_episode(
        tracker=tracker,
        event_audit=event_audit,
        pending_combat=pending_combat,
        ground_truth_audit=ground_truth_audit,
        event_lock=event_lock,
        episode_result=episode_result,
        policy_step=policy_step,
    )

    # No recovery/fallback/hit-reaction survives terminal.
    runtime.reset()

    return summary
