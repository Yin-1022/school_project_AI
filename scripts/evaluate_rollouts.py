from pathlib import Path
import numpy as np
from constant import ROLLOUT_DIR

def calculate_metrics(path):    
    rollout_name = Path(path).name
    episode_length = 0
    reward_sum = 0
    boss_hit_events = 0
    player_hit_events = 0
    action_switch_count = 0
    action_switch_ratio = 0
    visible_ratio = 0
    track_ratio = 0
    reacq_ratio = 0
    patrol_ratio = 0
    done_count = 0
    
    with np.load(path, allow_pickle=True) as data:
    
        final_action_id = data["final_action_id"]
        phase = data["phase"]
        visible = data["visible"]

        track_mask = (visible == 1) & (phase == "track")
        track_actions = final_action_id[track_mask]
        track_count = int(track_mask.sum())

        track_advance_ratio = action_ratio(track_actions, 1)      # Advance
        track_strafe_left_ratio = action_ratio(track_actions, 2)  # StrafeLeft
        track_strafe_right_ratio = action_ratio(track_actions, 3) # StrafeRight
        track_hold_ratio = action_ratio(track_actions, 0)         # Hold

        track_other_ratio = max(
            0.0,
            1.0
            - track_advance_ratio
            - track_strafe_left_ratio
            - track_strafe_right_ratio
            - track_hold_ratio
        )

        reacq_mask = (phase == "reacq")
        reacq_actions = final_action_id[reacq_mask]
        reacq_count = int(reacq_mask.sum())

        reacq_search_left_ratio = action_ratio(reacq_actions, 6)   # SearchTurnLeft
        reacq_search_right_ratio = action_ratio(reacq_actions, 7)  # SearchTurnRight
        reacq_patrol_left_ratio = action_ratio(reacq_actions, 8)   # PatrolStepLeft
        reacq_patrol_right_ratio = action_ratio(reacq_actions, 9)  # PatrolStepRight
        reacq_hold_ratio = action_ratio(reacq_actions, 0)          # Hold

        reacq_search_ratio = reacq_search_left_ratio + reacq_search_right_ratio
        reacq_patrol_ratio = reacq_patrol_left_ratio + reacq_patrol_right_ratio

        boss_hit = data["ue_boss_hit"]
        boss_hit_indices = np.where(boss_hit == 1)[0]

        pre_hit_actions = []
        window = 3

        for idx in boss_hit_indices:
            start = max(0, idx - window)
            pre_hit_actions.extend(final_action_id[start:idx].tolist())

        pre_hit_actions = np.array(pre_hit_actions, dtype=np.int64)

        pre_hit_advance_ratio = action_ratio(pre_hit_actions, 1)
        pre_hit_strafe_left_ratio = action_ratio(pre_hit_actions, 2)
        pre_hit_strafe_right_ratio = action_ratio(pre_hit_actions, 3)
        pre_hit_hold_ratio = action_ratio(pre_hit_actions, 0)
        pre_hit_search_ratio = action_ratio(pre_hit_actions, 6) + action_ratio(pre_hit_actions, 7)
        pre_hit_patrol_ratio = action_ratio(pre_hit_actions, 8) + action_ratio(pre_hit_actions, 9)
        pre_hit_evade_ratio = action_ratio(pre_hit_actions, 4) + action_ratio(pre_hit_actions, 5)

        episode_length = int(len(data["frame_id_end"]))
        reward_sum = float(data["reward"].sum())
        boss_hit_events = int(data["ue_boss_hit"].sum())
        player_hit_events = int(data["ue_player_hit"].sum())
        done_count = int(data["done"].sum())

        action_switch_count = int(np.sum(final_action_id[1:] != final_action_id[:-1]))
        action_switch_ratio = action_switch_count / episode_length

        visible_ratio = float(np.mean(visible))
        track_ratio = float(np.mean(phase == "track"))
        reacq_ratio = float(np.mean(phase == "reacq"))
        patrol_ratio = float(np.mean(phase == "patrol"))

        est_boss_damage_total = boss_hit_events * 26.67
        est_boss_damage_over_hp = est_boss_damage_total / 500
        est_player_damage_total = player_hit_events * 27.5
        est_player_damage_over_hp = est_player_damage_total / 100
        est_combat_diff_pct = est_player_damage_over_hp - est_boss_damage_over_hp
        boss_damage_per_100_steps = est_boss_damage_total / episode_length * 100
        player_damage_per_100_steps = est_player_damage_total / episode_length * 100
        est_boss_damage_over_hp_clipped = min(est_boss_damage_total / 500, 1.0)
        est_player_damage_over_hp_clipped = min(est_player_damage_total / 100, 1.0)

    return {
        "rollout_name" : rollout_name,
        "episode_length": episode_length,
        "reward_sum": reward_sum,
        "boss_hit_events": boss_hit_events,
        "player_hit_events": player_hit_events,
        "action_switch_count": action_switch_count,
        "action_switch_ratio": action_switch_ratio,
        "visible_ratio": visible_ratio,
        "track_ratio": track_ratio,
        "reacq_ratio": reacq_ratio,
        "patrol_ratio": patrol_ratio,
        "done_count": done_count,
        "est_boss_damage_total": est_boss_damage_total,
        "est_player_damage_total": est_player_damage_total,
        "est_combat_diff_pct": est_combat_diff_pct,
        "boss_damage_per_100_steps": boss_damage_per_100_steps,
        "player_damage_per_100_steps": player_damage_per_100_steps,
        "est_boss_damage_over_hp_clipped": est_boss_damage_over_hp_clipped,
        "est_player_damage_over_hp_clipped": est_player_damage_over_hp_clipped,
        "track_count": track_count,
        "track_advance_ratio": track_advance_ratio,
        "track_strafe_left_ratio": track_strafe_left_ratio,
        "track_strafe_right_ratio": track_strafe_right_ratio,
        "track_hold_ratio": track_hold_ratio,
        "track_other_ratio": track_other_ratio,

        "reacq_count": reacq_count,
        "reacq_search_ratio": reacq_search_ratio,
        "reacq_patrol_ratio": reacq_patrol_ratio,
        "reacq_hold_ratio": reacq_hold_ratio,

        "boss_hit_count": int(len(boss_hit_indices)),
        "pre_hit_advance_ratio": pre_hit_advance_ratio,
        "pre_hit_evade_ratio": pre_hit_evade_ratio,
        "pre_hit_hold_ratio": pre_hit_hold_ratio,
        "pre_hit_search_ratio": pre_hit_search_ratio,
        "pre_hit_patrol_ratio": pre_hit_patrol_ratio,
    }

def action_ratio(actions, action_id):
    if len(actions) == 0:
        return 0.0
    return float(np.mean(actions == action_id))

def summarize_metrics(metrics_list):
    if not metrics_list:
        return {}

    keys = [
        "episode_length",
        "reward_sum",
        "boss_hit_events",
        "player_hit_events",
        "est_boss_damage_total",
        "est_player_damage_total",
        "est_boss_damage_over_hp_clipped",
        "est_player_damage_over_hp_clipped",
        "est_combat_diff_pct",
        "boss_damage_per_100_steps",
        "player_damage_per_100_steps",
        "action_switch_count",
        "action_switch_ratio",
        "visible_ratio",
        "track_ratio",
        "reacq_ratio",
        "patrol_ratio",
        "done_count",
        "track_count",
        "track_advance_ratio",
        "track_strafe_left_ratio",
        "track_strafe_right_ratio",
        "track_hold_ratio",

        "reacq_count",
        "reacq_search_ratio",
        "reacq_patrol_ratio",
        "reacq_hold_ratio",

        "boss_hit_count",
        "pre_hit_advance_ratio",
        "pre_hit_evade_ratio",
        "pre_hit_hold_ratio",
    ]

    summary = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    return summary

def main():
    rollout_paths = sorted(ROLLOUT_DIR.glob("*.npz"))

    if not rollout_paths:
        print("No rollout files found.")
        return

    metrics_list = [calculate_metrics(path) for path in rollout_paths]

    print("=== Per-rollout metrics ===")
    for m in metrics_list:
        print(m)

    print("\n=== Summary ===")
    summary = summarize_metrics(metrics_list)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()