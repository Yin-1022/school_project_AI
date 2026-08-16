import numpy as np
import collections
from pathlib import Path
from reward_priority import (
    PRIORITY_NONE,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_HIGH,
    resolve_priority_reward,
)
from constant import ACTION_ID_TO_NAME

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
PRIORITY_NAMES = {
    0: "NONE",
    1: "LOW",
    2: "MED",
    3: "HIGH",
}

def dataset_info(files_len, total_transitions, files):
 print(f"===== Dataset =====\n"
          f"rollout files: {files_len}\n"
          f"total transitions: {total_transitions}\n"
          f"avg steps / rollout: {total_transitions / files_len:.1f}\n"
          f"min steps: {min(len(np.load(f)['frame_id_end']) for f in files)}\n"
          f"max steps: {max(len(np.load(f)['frame_id_end']) for f in files)}\n")

def priority_distribution(files_len, total_transitions, files):
    priority_counts = collections.Counter()
   
    for i in range(files_len):
        path = files[i]
        data = np.load(path,allow_pickle=False,)
        total_steps = len(data["frame_id_end"])
   
        for i in range(total_steps):
            resolved = resolve_priority_reward(
                reward_high=float(data["reward_high"][i]),
                reward_medium=float(data["reward_medium"][i]),
                reward_low=float(data["reward_low"][i]),
                ue_player_hit_count=int(data["ue_player_hit_count"][i]),
                ue_boss_hit_count=int(data["ue_boss_hit_count"][i]),
            )
            priority = resolved["priority"]
            priority_counts[priority] += 1
   
    print(f"===== Priority Distribution =====\n"
        f"NONE    {priority_counts[0]}   {priority_counts[0]/total_transitions*100:.1f}%\n"
        f"LOW     {priority_counts[1]}   {priority_counts[1]/total_transitions*100:.1f}%\n"
        f"MEDIUM  {priority_counts[2]}   {priority_counts[2]/total_transitions*100:.1f}%\n"
        f"HIGH    {priority_counts[3]}   {priority_counts[3]/total_transitions*100:.1f}%\n")

def selected_reward(files_len, files):
    low_counts = collections.Counter()
    medium_counts = collections.Counter()
    low_sum = 0.0
    medium_sum = 0.0

    for i in range(files_len):
        path = files[i]
        data = np.load(path,allow_pickle=False)
        total_steps = len(data["frame_id_end"])
   
        for i in range(total_steps):
            resolved = resolve_priority_reward(
                reward_high=float(data["reward_high"][i]),
                reward_medium=float(data["reward_medium"][i]),
                reward_low=float(data["reward_low"][i]),
                ue_player_hit_count=int(data["ue_player_hit_count"][i]),
                ue_boss_hit_count=int(data["ue_boss_hit_count"][i]),
            )
            priority = resolved["priority"]
            if priority == PRIORITY_LOW:
                if float(data["reward_low"][i]) > 0:
                    low_counts["positive"] += 1
                elif float(data["reward_low"][i]) < 0:
                    low_counts["negative"] += 1
                elif float(data["reward_low"][i]) == 0:
                    low_counts["zero"] += 1
                low_sum += float(data["reward_low"][i])
            elif priority == PRIORITY_MEDIUM:
                if float(data["reward_medium"][i]) > 0:
                    medium_counts["positive"] += 1
                elif float(data["reward_medium"][i]) < 0:
                    medium_counts["negative"] += 1
                elif float(data["reward_medium"][i]) == 0:
                    medium_counts["zero"] += 1
                medium_sum += float(data["reward_medium"][i])

    low_total = (
        low_counts["positive"]
        + low_counts["negative"]
        + low_counts["zero"]
    )

    medium_total = (
        medium_counts["positive"]
        + medium_counts["negative"]
        + medium_counts["zero"]
    )

    low_counts_mean = (
        low_sum / low_total
        if low_total > 0
        else 0.0
    )

    medium_counts_mean = (
        medium_sum / medium_total
        if medium_total > 0
        else 0.0
    )

    print(f"===== Selected Reward =====\n"
        f"LOW:\n"
        f"  positive: {low_counts['positive']}\n"
        f"  negative: {low_counts['negative']}\n"
        f"  zero: {low_counts['zero']}\n"
        f"  mean: {low_counts_mean:.3f}\n"
        f"MEDIUM:\n"
        f"  positive: {medium_counts['positive']}\n"
        f"  negative: {medium_counts['negative']}\n"
        f"  zero: {medium_counts['zero']}\n"
        f"  mean: {medium_counts_mean:.3f}\n")

def attack_event_distribution(files_len, files):
    attack_events = collections.Counter()

    for i in range(files_len):
        path = files[i]
        data = np.load(path,allow_pickle=False)
        attack_events["ue_att1_start"] += np.sum(data["ue_att1_start"])
        attack_events["ue_att1_end"] += np.sum(data["ue_att1_end"])
        attack_events["ue_att2_start"] += np.sum(data["ue_att2_start"])
        attack_events["ue_att2_end"] += np.sum(data["ue_att2_end"])
        attack_events["ue_player_hit_count"] += np.sum(data["ue_player_hit_count"])
        attack_events["ue_boss_hit_count"] += np.sum(data["ue_boss_hit_count"])

        total_steps = len(data["frame_id_end"])
           
        for j in range(total_steps):
            attack_events["both_hit_count"] += 1 if data["ue_player_hit_count"][j] > 0 and data["ue_boss_hit_count"][j] > 0 else 0

    print(f"===== Attack Event Distribution =====\n"
        f"Attack 1 Start: {attack_events['ue_att1_start']}\n"
        f"Attack 1 End: {attack_events['ue_att1_end']}\n"
        f"Attack 2 Start: {attack_events['ue_att2_start']}\n"
        f"Attack 2 End: {attack_events['ue_att2_end']}\n"
        f"Player gets hit count: {attack_events['ue_player_hit_count']}\n"
        f"Boss gets hit count: {attack_events['ue_boss_hit_count']}\n"
        f"Transitions with both hit: {attack_events['both_hit_count']}\n")

def action_distribution(files_len, files):
    action_counts = collections.Counter()
    action_positive = collections.Counter()
    action_zero = collections.Counter()
    action_negative = collections.Counter()

    action_reward_sum = collections.Counter()

    for i in range(files_len):
        path = files[i]
        data = np.load(path,allow_pickle=False)
        total_steps = len(data["frame_id_end"])

        for j in range(total_steps):
            action_id = int(data["final_action_id"][j])
            action_counts[action_id] += 1

            resolved = resolve_priority_reward(
                reward_high=float(data["reward_high"][j]),
                reward_medium=float(data["reward_medium"][j]),
                reward_low=float(data["reward_low"][j]),
                ue_player_hit_count=int(data["ue_player_hit_count"][j]),
                ue_boss_hit_count=int(data["ue_boss_hit_count"][j]),
            )

            selected_reward = resolved["reward"]

            if selected_reward > 0:
                action_positive[action_id] += 1
            elif selected_reward == 0:
                action_zero[action_id] += 1
            else:
                action_negative[action_id] += 1

            action_reward_sum[action_id] += selected_reward

    print(f"===== Action Distribution =====\n"
          f"Action               N       %    Reward+    Reward0     Reward-     Mean selected reward")
    for action_id, count in action_counts.items():
        action_name = ACTION_ID_TO_NAME.get(action_id, f"Action {action_id}")
        action_percentage = count / sum(action_counts.values()) * 100
        reward_positive = action_positive[action_id]
        reward_zero = action_zero[action_id]
        reward_negative = action_negative[action_id]

        mean_selected_reward = (
            action_reward_sum[action_id] / count
            if count > 0
            else 0.0
        )

        print(f"{action_name:<15} {count:>6} {action_percentage:>6.2f}% {reward_positive:>10} {reward_zero:>10} {reward_negative:>10} {mean_selected_reward:>15.2f}")

def proposeToFinalCount(files_len, files):
    action_changing_counts = collections.Counter()

    same_count = 0
    total_count = 0

    for i in range(files_len):
        path = files[i]
        data = np.load(path,allow_pickle=False)
        total_steps = len(data["frame_id_end"])

        for j in range(total_steps):
            proposed_action_id = data["proposed_action_id"][j]
            final_action_id = data["final_action_id"][j]

            total_count += 1

            if proposed_action_id == final_action_id:
                same_count += 1
            else:
                action_changing_counts[(proposed_action_id, final_action_id)] += 1

        changed_count = total_count - same_count
        change_rate = (changed_count / total_count) * 100 if total_count > 0 else 0.0

    print(f"\n===== Postprocess =====\n"
              f"Same: {same_count}\n"
              f"Changed: {changed_count}\n"
              f"Change rate: {change_rate:.2f}%\n")
    print(f"Top Postprocess Changes")
    for (proposed_action_id, final_action_id), count in action_changing_counts.most_common(10):
        proposed_action_name = ACTION_ID_TO_NAME.get(proposed_action_id, f"Action {proposed_action_id}")
        final_action_name = ACTION_ID_TO_NAME.get(final_action_id, f"Action {final_action_id}")
        print(f" {proposed_action_name} -> {final_action_name}: {count}")
            
def phase_distribution(files_len, files):
    phase_counts = collections.Counter()
    action_counts = collections.Counter()
    phase_action_counts = collections.Counter()

    for i in range(files_len):

        path = files[i]
        data = np.load(path,allow_pickle=False)
        total_steps = len(data["frame_id_end"])

        for j in range(total_steps):
            phase = str(data["phase"][j])
            action_id = int(data["final_action_id"][j])

            phase_counts[phase] += 1
            phase_action_counts[(phase, action_id)] += 1

    phase_total = sum(phase_counts.values())

    for phase in ["track", "reacq", "patrol"]:
        print(phase)

        for (pair_phase, action_id), count in phase_action_counts.items():
            if pair_phase != phase:
                continue

            action_name = ACTION_ID_TO_NAME.get(
                action_id,
                f"Action {action_id}",
            )

            print(f" {action_name}: {count}")

def sanity_check(files):
    medium_mismatch = 0

    att1_start = 0
    att1_end = 0

    att2_start = 0
    att2_end = 0

    for path in files:
        data = np.load(path, allow_pickle=False)

        total_steps = len(data["frame_id_end"])
        att1_start += int(
            np.sum(data["ue_att1_start"])
        )
    
        att1_end += int(
            np.sum(data["ue_att1_end"])
        )

        att2_start += int(
            np.sum(data["ue_att2_start"])
        )

        att2_end += int(
            np.sum(data["ue_att2_end"])
        )

        for j in range(total_steps):
            expected_medium = (
                int(data["ue_player_hit_count"][j])
                - int(data["ue_boss_hit_count"][j])
            )

            actual_medium = float(
                data["reward_medium"][j]
            )

            if not np.isclose(
                expected_medium,
                actual_medium,
            ):
                medium_mismatch += 1

    print("\n===== Sanity Check =====")
    print(
        f"Medium reward mismatch: "
        f"{medium_mismatch}"
    )

    print(
        f"ATT1 start/end mismatch: "
        f"{att1_start - att1_end}"
    )

    print(
        f"ATT2 start/end mismatch: "
        f"{att2_start - att2_end}"
    )

def main() -> None:
    files = sorted(
        ROLLOUT_DIR.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(f"No rollout files found in {ROLLOUT_DIR}")

    files_len = len(files)
    total_transitions = sum(len(np.load(f)['frame_id_end']) for f in files)

    dataset_info(files_len, total_transitions, files)
    priority_distribution(files_len, total_transitions, files)
    selected_reward(files_len, files)
    attack_event_distribution(files_len, files)
    action_distribution(files_len, files)
    proposeToFinalCount(files_len, files)
    phase_distribution(files_len, files)
    sanity_check(files)
if __name__ == "__main__":
    main()