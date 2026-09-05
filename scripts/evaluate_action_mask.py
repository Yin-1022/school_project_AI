import torch
import numpy as np

from constant import ACTION_ID_TO_NAME, ROLLOUT_DIR

ACTION_NAME_TO_ID = {
    name: action_id
    for action_id, name
    in ACTION_ID_TO_NAME.items()
}

semantic_action_ids = [
    ACTION_NAME_TO_ID[
        "SearchTurnLeft"
    ],
    ACTION_NAME_TO_ID[
        "SearchTurnRight"
    ],
    ACTION_NAME_TO_ID[
        "PatrolStepLeft"
    ],
    ACTION_NAME_TO_ID[
        "PatrolStepRight"
    ],
]

def safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(values.mean())

def analyze_rollout_group(files, has_action_mask):
    num_actions = len(ACTION_ID_TO_NAME)

    transition_count = 0
    intervention_count = 0

    proposed_count = np.zeros(num_actions, dtype=np.int64)
    final_count = np.zeros(num_actions, dtype=np.int64)

    phase_counts = {
        "track": 0,
        "reacq": 0,
        "patrol": 0,
    }

    phase_intervention_counts = {
        "track": 0,
        "reacq": 0,
        "patrol": 0,
    }

    visible_track_count = 0
    visible_track_intervention_count = 0

    # Post-Mask only
    mask_active_count = 0
    masked_slot_count = 0

    masked_action_counts = np.zeros(num_actions, dtype=np.int64)
    masked_mass_sum = 0.0
    masked_mass_active_sum = 0.0
    max_masked_mass = 0.0

    intervention_masked_count = 0
    intervention_unmasked_count = 0

    masked_transition_count = 0
    unmasked_transition_count = 0

    semantic_transition_count = 0
    semantic_raw_mass_sum = 0.0

    #File reading
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "action_mask_mode" not in data.files:
                continue

            if data["map_version"].item() != "new_map_v1":
                continue

            if (
                data["action_space_version"].item()
                != "no_retreat_v1"
            ):
                continue

            mode = data["action_mask_mode"].item()

            if mode == "baseline":
                baseline_files.append(path)

            elif mode == "masked":
                masked_files.append(path)


            proposed = data["proposed_action_id"].astype(np.int64)
            final = data["final_action_id"].astype(np.int64)
            n = len(proposed)

            transition_count += n

            intervened = proposed != final
            intervention_count += intervened.sum()

            proposed_count += np.bincount(proposed, minlength=num_actions)
            final_count += np.bincount(final, minlength=num_actions)

            phase = data["phase"]
            for phase_name in phase_counts:
                phase_mask = phase == phase_name
                phase_counts[phase_name] += int(phase_mask.sum())
                phase_intervention_counts[phase_name] += int(intervened[phase_mask].sum())

            visible = data["visible"]
            visible_track = ((phase == "track")& (visible == 1))
            visible_track_count += int(visible_track.sum())
            visible_track_intervention_count += int(intervened[visible_track].sum())

            if has_action_mask:
                action_mask = data["action_mask"].astype(bool)

                illegal = ~action_mask
                mask_active = illegal.any(axis=1)

                current_mask_active_count = int(mask_active.sum())
                mask_active_count += current_mask_active_count

                unmasked_transition_count += (n-current_mask_active_count)

                masked_slot_count += int(illegal.sum())
                masked_action_counts += illegal.sum(axis=0)

                # Raw policy probabilities
                logits = torch.from_numpy(data["logits"].astype(np.float32))
                raw_probs = torch.softmax(logits, dim=-1).numpy()

                # Masked Probability Mass
                masked_prob_mass = (raw_probs * illegal).sum(axis=1)
                masked_mass_sum += float(masked_prob_mass.sum())
                masked_mass_active_sum += float(masked_prob_mass[mask_active].sum())
                if len(masked_prob_mass) > 0:
                    max_masked_mass = max(max_masked_mass, float(masked_prob_mass.max()))

                # Intervention when mask is active/inactive
                intervention_masked_count += int(intervened[mask_active].sum())
                intervention_unmasked_count += int(intervened[~mask_active].sum())

                # Visible-track semantic stats
                semantic_active = (phase == "track") & (visible == 1)
                semantic_count = int(semantic_active.sum())
                semantic_transition_count += semantic_count

                # Raw probabilities for Search/Patrol
                semantic_raw_mass = (raw_probs[:, semantic_action_ids].sum(axis=1))
                semantic_raw_mass_sum += float(semantic_raw_mass[semantic_active].sum())

    if transition_count == 0:
        raise ValueError("No transitions found.")

    intervention_rate = intervention_count / transition_count
    proposed_distribution = proposed_count / transition_count
    final_distribution = final_count / transition_count
    phase_distribution = {
        phase_name: count / transition_count
        for phase_name, count in phase_counts.items()
    }
    phase_intervention_rates = {
        phase_name: (
            phase_intervention_counts[phase_name]/ phase_counts[phase_name]
            if phase_counts[phase_name] > 0 else 0.0
        )
        for phase_name in phase_counts
    }
    visible_track_intervention_rate = (
        visible_track_intervention_count/ visible_track_count
        if visible_track_count > 0 else 0.0
    )

    if has_action_mask:
        mask_active_rate = mask_active_count / transition_count
        avg_masked_actions = masked_slot_count / transition_count
        avg_masked_actions_when_active = masked_slot_count / mask_active_count if mask_active_count > 0 else 0.0
        mean_masked_mass = masked_mass_sum / transition_count
        mean_masked_mass_when_active = masked_mass_active_sum / mask_active_count if mask_active_count > 0 else 0.0

        intervention_when_masked = intervention_masked_count / mask_active_count if mask_active_count > 0 else 0.0
        intervention_when_unmasked = intervention_unmasked_count / unmasked_transition_count if unmasked_transition_count > 0 else 0.0

        semantic_rate = semantic_transition_count / transition_count
        mean_semantic_raw_mass = semantic_raw_mass_sum / semantic_transition_count if semantic_transition_count > 0 else 0.0

    result = {
        "file_count": len(files),
        "transition_count": transition_count,
        "intervention_count": intervention_count,
        "intervention_rate": intervention_rate,
        "proposed_count": proposed_count,
        "final_count": final_count,
        "proposed_distribution": proposed_distribution,
        "final_distribution": final_distribution,
        "phase_counts": phase_counts,
        "phase_distribution": phase_distribution,
        "phase_intervention_counts": phase_intervention_counts,
        "phase_intervention_rates": phase_intervention_rates,
        "visible_track_count": visible_track_count,
        "visible_track_intervention_count": visible_track_intervention_count,
        "visible_track_intervention_rate": visible_track_intervention_rate,
    }

    if has_action_mask:
        result.update({
            "mask_active_count": mask_active_count,
            "mask_active_rate": mask_active_rate,
            "masked_slot_count": masked_slot_count,
            "avg_masked_actions": avg_masked_actions,
            "avg_masked_actions_when_active": avg_masked_actions_when_active,
            "masked_action_counts": masked_action_counts,
            "mean_masked_mass": mean_masked_mass,
            "mean_masked_mass_when_active": mean_masked_mass_when_active,
            "max_masked_mass": max_masked_mass,
            "intervention_when_masked": intervention_when_masked,
            "intervention_when_unmasked": intervention_when_unmasked,
            "semantic_transition_count": semantic_transition_count,
            "semantic_rate": semantic_rate,
            "mean_semantic_raw_mass": mean_semantic_raw_mass,
        })

    return result

def main() -> None:
    rollout_files = sorted(
        ROLLOUT_DIR.glob("*.npz")
    )

    pre_mask_files = []
    post_mask_files = []
    
    print(f"loading: {ROLLOUT_DIR}")
    
    for path in rollout_files:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            if "action_mask" in data.files:
                post_mask_files.append(path)
            else:
                pre_mask_files.append(path)

    print("\nPRE files:")
    for path in pre_mask_files: 
        with np.load(path, allow_pickle=False) as data:
            print(
                path.name,
                len(data["proposed_action_id"]),
            )

    if not pre_mask_files:
        raise RuntimeError(
            "No pre-mask rollouts found"
        )

    if not post_mask_files:
        raise RuntimeError(
            "No post-mask rollouts found"
        )

    pre_stats = analyze_rollout_group(pre_mask_files, has_action_mask=False)
    post_stats = analyze_rollout_group(post_mask_files, has_action_mask=True)


    print(f"--- Action Mask Comparison ---\n")
    print("Dataset")
    print("                    PRE   POST")
    print(f"Files             {pre_stats['file_count']:>5}  {post_stats['file_count']:>5}")
    print(f"Transitions       {pre_stats['transition_count']:>5}  {post_stats['transition_count']:>5}")

    delta_pp = (post_stats["intervention_rate"] - pre_stats["intervention_rate"]) * 100
    print("\nPostprocess")
    print("                      PRE    POST")
    print(f"Interventions       {pre_stats['intervention_count']:>5}   {post_stats['intervention_count']:>5}")
    print(f"Intervention Rate  {pre_stats['intervention_rate']:.4f}  {post_stats['intervention_rate']:.4f}")
    print(f"Change                     {delta_pp:+.2f} pp")

    print("\n--- Proposed Action Distribution ---\n")
    print("Action Name            PRE       POST")
    for action_id, action_name in ACTION_ID_TO_NAME.items():
        pre_dist = pre_stats["proposed_distribution"][action_id]
        post_dist = post_stats["proposed_distribution"][action_id]
        print(f"{action_name:<20} {pre_dist:>7.2%}  {post_dist:>7.2%}")

    print("\n--- Final Action Distribution ---\n")
    print("Action Name            PRE       POST")

    for action_id, action_name in ACTION_ID_TO_NAME.items():
        pre_dist = pre_stats["final_distribution"][action_id]
        post_dist = post_stats["final_distribution"][action_id]
        print(
            f"{action_name:<20} "
            f"{pre_dist:>7.2%}  "
            f"{post_dist:>7.2%}"
        )

    print("\n--- Phase Distribution ---\n")
    print("Phase               PRE     POST")
    for phase_name, pre_dist in pre_stats["phase_distribution"].items():
        post_dist = post_stats["phase_distribution"][phase_name]
        print(f"{phase_name:<15} {pre_dist:>7.2%}  {post_dist:>7.2%}")

    print("\n--- Phase Intervention Rates ---\n")
    print("Phase               PRE     POST")
    for phase_name, pre_rate in pre_stats["phase_intervention_rates"].items():
        post_rate = post_stats["phase_intervention_rates"][phase_name]
        print(f"{phase_name:<15} {pre_rate:>7.2%}  {post_rate:>7.2%}")

    print("\n--- Visible-Track Intervention Rate ---\n")
    print(
        "PRE:  "
        f"{pre_stats['visible_track_intervention_count']}"
        "/"
        f"{pre_stats['visible_track_count']} "
        f"({pre_stats['visible_track_intervention_rate']:.2%})"
    )

    print(
        "POST: "
        f"{post_stats['visible_track_intervention_count']}"
        "/"
        f"{post_stats['visible_track_count']} "
        f"({post_stats['visible_track_intervention_rate']:.2%})"
    )

    print("\n--- Post-mask Diagnostics ---\n")
    print(f"Mask active rate: {post_stats['mask_active_rate']:.4f}")
    print(f"Avg masked actions when active: {post_stats['avg_masked_actions_when_active']:.4f}")
    print(f"Mean masked probability mass: {post_stats['mean_masked_mass']:.4f}")
    print(f"Mean masked mass when active: {post_stats['mean_masked_mass_when_active']:.4f}")
    print(f"Max masked mass: {post_stats['max_masked_mass']:.4f}")

    print(f"Intervention | mask active: {post_stats['intervention_when_masked']:.4f}")
    print(f"Intervention | mask inactive: {post_stats['intervention_when_unmasked']:.4f}")

    print(f"Visible-track rate: {post_stats['semantic_rate']:.4f}")
    print(f"Raw Search/Patrol mass: {post_stats['mean_semantic_raw_mass']:.4f}")
    
if __name__ == "__main__":
    main()