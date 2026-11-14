
from episode_metrics import (
    EpisodeEventAudit,
    PendingCombatEvents,
    format_summary,
)
from ground_truth_audit import GroundTruthAudit, format_gt_summary
from transition_tracker import TransitionTracker


def capture_terminal_combat_state(
    *,
    event_lock,
    event_audit: EpisodeEventAudit,
    pending_combat: PendingCombatEvents,
) -> dict:
    """Consume pending combat and freeze raw UE event counters."""

    with event_lock:
        pending_boss_hits = pending_combat.boss_hit_count
        pending_player_hits = pending_combat.player_hit_count
        pending_boss_damage = pending_combat.boss_damage
        pending_player_damage = pending_combat.player_damage

        pending_combat.reset()

        raw_player_hits_on_boss = event_audit.player_hits_on_boss_raw
        raw_boss_hits_on_player = event_audit.boss_hits_on_player_raw

        boss_damage_to_player_raw = event_audit.boss_damage_to_player_raw
        player_damage_to_boss_raw = event_audit.player_damage_to_boss_raw
        final_player_health = event_audit.final_player_health
        final_boss_health = event_audit.final_boss_health

    return {
        "pending_boss_hits": pending_boss_hits,
        "pending_player_hits": pending_player_hits,
        "pending_boss_damage": pending_boss_damage,
        "pending_player_damage": pending_player_damage,
        "raw_player_hits_on_boss": raw_player_hits_on_boss,
        "raw_boss_hits_on_player": raw_boss_hits_on_player,
        "boss_damage_to_player_raw": boss_damage_to_player_raw,
        "player_damage_to_boss_raw": player_damage_to_boss_raw,
        "final_player_health": final_player_health,
        "final_boss_health": final_boss_health,
    }


def build_terminal_event_audit_snapshot(
    *,
    event_audit: EpisodeEventAudit,
    terminal_combat_state: dict,
) -> dict:
    """Build the existing EpisodeMetrics audit snapshot."""

    state = terminal_combat_state

    return {
        "player_hits_on_boss_raw": state["raw_player_hits_on_boss"],
        "boss_hits_on_player_raw": state["raw_boss_hits_on_player"],
        "unattributed_boss_hits_on_player":
            event_audit.unattributed_boss_hits_on_player,
        "unattributed_player_hits_on_boss":
            event_audit.unattributed_player_hits_on_boss,
        "unattributed_boss_damage_to_player":
            event_audit.unattributed_boss_damage_to_player,
        "unattributed_player_damage_to_boss":
            event_audit.unattributed_player_damage_to_boss,
        "boss_damage_to_player_raw": state["boss_damage_to_player_raw"],
        "player_damage_to_boss_raw": state["player_damage_to_boss_raw"],
        "final_player_health": state["final_player_health"],
        "final_boss_health": state["final_boss_health"],
        "player_max_hp": event_audit.player_max_hp,
        "boss_max_hp": event_audit.boss_max_hp,
    }


def finalize_episode(
    *,
    tracker: TransitionTracker,
    event_audit: EpisodeEventAudit,
    pending_combat: PendingCombatEvents,
    ground_truth_audit: GroundTruthAudit,
    event_lock,
    episode_result: int,
    policy_step: int,
) -> dict:
    """Finalize a terminal episode using the existing reward semantics."""

    terminal_combat_state = capture_terminal_combat_state(
        event_lock=event_lock,
        event_audit=event_audit,
        pending_combat=pending_combat,
    )

    tracker.record_terminal_combat(terminal_combat_state)

    event_audit_snapshot = build_terminal_event_audit_snapshot(
        event_audit=event_audit,
        terminal_combat_state=terminal_combat_state,
    )

    with event_lock:
        if ground_truth_audit.current is not None:
            ground_truth_audit.abort_action("episode_terminal")

    tracker.finalize_transitions(episode_result)

    summary = tracker.episode_metrics.finish_episode(
        episode_result=episode_result,
        event_audit=event_audit_snapshot,
        policy_step=policy_step,
    )

    print(format_summary(summary))
    print(format_gt_summary(ground_truth_audit.build_summary()))

    tracker.flush()

    return summary
