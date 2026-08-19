from pathlib import Path
import collections
import numpy as np
from reward_priority import resolve_priority_reward
from constant import ACTION_ID_TO_NAME
PRIORITY_NAMES = {
    0: "NONE",
    1: "LOW",
    2: "MED",
    3: "HIGH",
}

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

def get_source(proposed_action_id: int,final_action_id: int) -> str:
    if proposed_action_id == final_action_id:
        return "DIRECT"

    return "CORRECTED"

def get_outcome(selected_reward: float) -> str:
    if selected_reward > 0:
        return "POSITIVE"

    if selected_reward < 0:
        return "NEGATIVE"

    return "NEUTRAL"

def analyze_training_targets():
    source_outcome_counts = collections.Counter()
    corrected_pair_counts = collections.Counter()
    learning_action_counts = collections.Counter()
    corrected_pairs = set()

    files = sorted(ROLLOUT_DIR.glob("*.npz"))

    for path in files:
        data = np.load(
            path,
            allow_pickle=False,
        )

        total_steps = len(data["frame_id_end"])

        for i in range(total_steps):
            proposed_action_id = int(data["proposed_action_id"][i])
            final_action_id = int(data["final_action_id"][i])
            learning_action_id = final_action_id

            resolved = resolve_priority_reward(
                reward_high=float(
                    data["reward_high"][i]
                ),
                reward_medium=float(
                    data["reward_medium"][i]
                ),
                reward_low=float(
                    data["reward_low"][i]
                ),
                ue_player_hit_count=int(
                    data["ue_player_hit_count"][i]
                ),
                ue_boss_hit_count=int(
                    data["ue_boss_hit_count"][i]
                ),
            )

            selected_reward = resolved["reward"]
            priority = resolved["priority"]
            priority_name = PRIORITY_NAMES[priority]

            source = get_source(
                proposed_action_id,
                final_action_id,
            )

            outcome = get_outcome(
                selected_reward
            )

            source_outcome_counts[(source, priority_name, outcome)] += 1

            if source == "CORRECTED":
                corrected_pair_counts[(proposed_action_id,final_action_id,priority_name,outcome)] += 1
                corrected_pairs.add((proposed_action_id,final_action_id, priority))

            learning_action_counts[(learning_action_id,priority_name,outcome)] += 1

    print(f"===== Training Target Source =====\n"
          f"DIRECT\n"
          f" Low\n"
          f"  Positive: {source_outcome_counts[('DIRECT', 'LOW', 'POSITIVE')]}\n" 
          f"  Negative: {source_outcome_counts[('DIRECT', 'LOW', 'NEGATIVE')]}\n"
          f"  Neutral:  {source_outcome_counts[('DIRECT', 'LOW', 'NEUTRAL')]} \n"
          f" Medium\n"
          f"  Positive: {source_outcome_counts[('DIRECT', 'MED', 'POSITIVE')]}\n"
          f"  Negative: {source_outcome_counts[('DIRECT', 'MED', 'NEGATIVE')]}\n"
          f"  Neutral:  {source_outcome_counts[('DIRECT', 'MED', 'NEUTRAL')]} \n"
          f" High\n"
          f"  Positive: {source_outcome_counts[('DIRECT', 'HIGH', 'POSITIVE')]}\n"
          f"  Negative: {source_outcome_counts[('DIRECT', 'HIGH', 'NEGATIVE')]}\n"
          f"  Neutral:  {source_outcome_counts[('DIRECT', 'HIGH', 'NEUTRAL')]} \n"
          f"CORRECTED\n"
          f" Low\n"
          f"  Positive: {source_outcome_counts[('CORRECTED', 'LOW', 'POSITIVE')]}\n" 
          f"  Negative: {source_outcome_counts[('CORRECTED', 'LOW', 'NEGATIVE')]}\n"
          f"  Neutral:  {source_outcome_counts[('CORRECTED', 'LOW', 'NEUTRAL')]} \n"
          f" Medium\n"
          f"  Positive: {source_outcome_counts[('CORRECTED', 'MEDIUM', 'POSITIVE')]}\n"
          f"  Negative: {source_outcome_counts[('CORRECTED', 'MEDIUM', 'NEGATIVE')]}\n"
          f"  Neutral:  {source_outcome_counts[('CORRECTED', 'MEDIUM', 'NEUTRAL')]} \n"
          f" High\n"
          f"  Positive: {source_outcome_counts[('CORRECTED', 'HIGH', 'POSITIVE')]}\n"
          f"  Negative: {source_outcome_counts[('CORRECTED', 'HIGH', 'NEGATIVE')]}\n"
          f"  Neutral:  {source_outcome_counts[('CORRECTED', 'HIGH', 'NEUTRAL')]} \n")

    print(f"\n===== Corrected Pair Outcomes =====")
    print(
        f"{'Proposed':<15} "
        f"{'Final':<10} "
        f"{'+':>7} "
        f"{'0':>7} "
        f"{'-':>7}"
    )
    for proposed_id, final_id, priority in corrected_pairs:
        priority_name = PRIORITY_NAMES.get(priority, "UNKNOWN")
        positive = corrected_pair_counts[(proposed_id, final_id, priority_name, "POSITIVE")]
        neutral = corrected_pair_counts[(proposed_id, final_id, priority_name, "NEUTRAL")]
        negative = corrected_pair_counts[(proposed_id, final_id, priority_name, "NEGATIVE")]

        print(f"{ACTION_ID_TO_NAME[proposed_id]:<15} {ACTION_ID_TO_NAME[final_id]:<16} {positive:<7} {neutral:<7} {negative:<7}")

    print(f"\n===== Learning Action Outcomes =====")
    print(f"Action               Positive   Neutral   Negative")
    for action_id in range(len(ACTION_ID_TO_NAME)):
        positive = sum(
            learning_action_counts[
                (
                    action_id,
                    priority_name,
                    "POSITIVE",
                )
            ]
            for priority_name in PRIORITY_NAMES.values()
        )

        neutral = sum(
            learning_action_counts[
                (
                    action_id,
                    priority_name,
                    "NEUTRAL",
                )
            ]
            for priority_name in PRIORITY_NAMES.values()
        )

        negative = sum(
            learning_action_counts[
                (
                    action_id,
                    priority_name,
                    "NEGATIVE",
                )
            ]
            for priority_name in PRIORITY_NAMES.values()
        )

        print(
            f"{ACTION_ID_TO_NAME[action_id]:<20} "
            f"{positive:<10} "
            f"{neutral:<10} "
            f"{negative:<10}"
        )

def main():
    analyze_training_targets()

if __name__ == "__main__":
    main()
