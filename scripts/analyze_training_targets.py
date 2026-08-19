from pathlib import Path
import collections
import numpy as np
from reward_priority import resolve_priority_reward
from constant import ACTION_ID_TO_NAME

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

            source = get_source(
                proposed_action_id,
                final_action_id,
            )

            outcome = get_outcome(
                selected_reward
            )

            source_outcome_counts[(source, outcome)] += 1

            if source == "CORRECTED":
                corrected_pair_counts[
                    (
                        proposed_action_id,
                        final_action_id,
                        outcome,
                    )
                ] += 1

                corrected_pairs.add((proposed_action_id,final_action_id,))

            learning_action_counts[(learning_action_id,outcome)] += 1

    print(f"===== Training Target Source =====\n"
          f"DIRECT\n"
          f" Positive: {source_outcome_counts[('DIRECT', 'POSITIVE')]}\n" 
          f" Negative: {source_outcome_counts[('DIRECT', 'NEGATIVE')]}\n"
          f" Neutral:  {source_outcome_counts[('DIRECT', 'NEUTRAL')]} \n"
          f"CORRECTED\n"
          f" Positive: {source_outcome_counts[('CORRECTED', 'POSITIVE')]}\n" 
          f" Negative: {source_outcome_counts[('CORRECTED', 'NEGATIVE')]}\n"
          f" Neutral:  {source_outcome_counts[('CORRECTED', 'NEUTRAL')]} ")

    

    if source == "CORRECTED":
        corrected_pairs.add(
            (
                proposed_action_id,
                final_action_id,
            )
        )

    print(f"\n===== Corrected Pair Outcomes =====")
    print(
        f"{'Proposed':<15} "
        f"{'Final':<10} "
        f"{'+':>7} "
        f"{'0':>7} "
        f"{'-':>7}"
    )
    for proposed_id, final_id in corrected_pairs:
        positive = corrected_pair_counts[(proposed_id, final_id, "POSITIVE")]
        neutral = corrected_pair_counts[(proposed_id, final_id, "NEUTRAL")]
        negative = corrected_pair_counts[(proposed_id, final_id, "NEGATIVE")]

        print(f"{ACTION_ID_TO_NAME[proposed_id]:<15} {ACTION_ID_TO_NAME[final_id]:<16} {positive:<7} {neutral:<7} {negative:<7}")

    print(f"\n===== Learning Action Outcomes =====")
    print(f"Action               Positive   Neutral   Negative")
    for action_id in range(len(ACTION_ID_TO_NAME)):
        positive = learning_action_counts[(action_id, "POSITIVE")]
        neutral = learning_action_counts[(action_id, "NEUTRAL")]
        negative = learning_action_counts[(action_id, "NEGATIVE")]

        print(f"{ACTION_ID_TO_NAME[action_id]:<20} {positive:<10} {neutral:<10} {negative:<10}")

def main():
    analyze_training_targets()

if __name__ == "__main__":
    main()
