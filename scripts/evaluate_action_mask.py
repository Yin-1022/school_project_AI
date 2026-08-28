import torch
import numpy as np
from pathlib import Path

from constant import ACTION_ID_TO_NAME
from check_masked_impala import find_latest_masked_rollout

BATCH_UNROLLS = 1
ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

ACTION_NAME_TO_ID = {
    name: action_id
    for action_id, name
    in ACTION_ID_TO_NAME.items()
}

def safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(values.mean())

def main() -> None:
    path = find_latest_masked_rollout()
    
    print(f"loading: {path}")
    
    data = np.load(
        path,
        allow_pickle=False,
    )

    # Intervention rate
    proposed = data["proposed_action_id"]
    final = data["final_action_id"]

    intervened = proposed != final
    intervention_rate = intervened.mean()

    # Mask active rate
    action_mask = data["action_mask"].astype(bool)
    mask_active = (~action_mask).any(axis=1)

    mask_active_rate = mask_active.mean()

    # How many actions masked per timestep
    masked_action_count = (~action_mask).sum(axis=1)
    avg_masked_action_count = masked_action_count.mean()
    avg_masked_when_active = masked_action_count[mask_active].mean()

    # Masked Probability Mass
    logits = torch.from_numpy(data["logits"]).float()
    raw_probs = torch.softmax(logits, dim=-1).numpy()

    illegal = ~action_mask
    masked_prob_mass = (raw_probs * illegal).sum(axis=1)

    mean_masked_mass = masked_prob_mass.mean()
    mean_masked_mass_when_active = masked_prob_mass[mask_active].mean()
    max_masked_mass = masked_prob_mass.max()

    # Masked Action Distribution
    mask_count = (~action_mask).sum(axis=0)
    mask_count_dict = {
        ACTION_ID_TO_NAME[i]: int(mask_count[i])
        for i in range(len(mask_count))
    }

    # Intevention when mask is active/inactive
    intervention_when_masked = safe_mean(
        intervened[mask_active]
    )

    intervention_when_unmasked = safe_mean(
        intervened[~mask_active]
    )

    # Check visible + track semantic mask
    visible = data["visible"]
    phase = data["phase"]

    semantic_active = (phase == "track") & (visible == 1)
    semantic_rate = semantic_active.mean()
    semantic_masked_mass = masked_prob_mass[semantic_active].mean()

    # Raw semantic actions
    semantic_action_ids = [
        ACTION_NAME_TO_ID["SearchTurnLeft"],
        ACTION_NAME_TO_ID["SearchTurnRight"],
        ACTION_NAME_TO_ID["PatrolStepLeft"],
        ACTION_NAME_TO_ID["PatrolStepRight"],
    ]
    semantic_raw_mass = (raw_probs[:, semantic_action_ids].sum(axis=1))
    mean_semantic_raw_mass = safe_mean(semantic_raw_mass[semantic_active])


    print(f"--- Postprocess ---")
    print(f"Invenvention: {intervened.sum()} / {len(intervened)}")
    print(f"Intervention Rate: {intervention_rate:.4f}")

    print(f"\n--- Mask Activity ---")
    print(f"Mask Active Rate: {mask_active_rate:.4f}")
    print(f"Avg Masked Action Count: {avg_masked_action_count:.4f}")
    print(f"Avg Masked Action Count When Active: {avg_masked_when_active:.4f}")

    print(f"\n--- Probability Mass ---")
    print(f"Mean Masked Probability Mass: {mean_masked_mass:.4f}")
    print(f"Mean Masked Probability Mass When Active: {mean_masked_mass_when_active:.4f}")
    print(f"Max Masked Probability Mass: {max_masked_mass:.4f}")

    print(f"\n--- Mask By Action ---")
    print(f"{mask_count_dict}")

    print(f"\n--- Intervention By Mask State ---")
    print(f"Masked active: {intervention_when_masked:.4f}")
    print(f"Masked inactive: {intervention_when_unmasked:.4f}")

    print(f"\n--- Mean total masked mass during visible-track ---")
    print(f"Mean masked mass during visible-track: {semantic_masked_mass:.4f}")

    print(f"\n--- Visible Track Semantic Mask ---")
    print(f"Visible-track transitions: {semantic_active.sum()}")
    print(f"Visible-track rate: {semantic_rate:.4f}")
    print(f"Raw Search/Patrol probability mass: {mean_semantic_raw_mass:.4f}")
    
if __name__ == "__main__":
    main()