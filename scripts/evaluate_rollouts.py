from pathlib import Path
import numpy as np

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
        "est_player_damage_over_hp_clipped": est_player_damage_over_hp_clipped
    }

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
        "visible_ratio",
        "track_ratio",
        "reacq_ratio",
        "patrol_ratio",
        "done_count",
    ]

    summary = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    return summary

def main():
    rollout_dir = Path("data/rollouts")
    rollout_paths = sorted(rollout_dir.glob("*.npz"))

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