import numpy as np
from pathlib import Path
from reward_priority import resolve_priority_reward
from scripts.constant import ACTION_ID_TO_NAME

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
PRIORITY_NAMES = {
    0: "NONE",
    1: "LOW",
    2: "MED",
    3: "HIGH",
}
    
def main() -> None:
    files = sorted(
        ROLLOUT_DIR.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(
            f"No rollout files found in {ROLLOUT_DIR}"
        )

    path = files[-1]

    print(f"loading: {path}")

    data = np.load(
        path,
        allow_pickle=False,
    )

    print("\n===== NPZ Keys =====")

    for key in data.files:
        print(
            f"{key:<24} "
            f"shape={data[key].shape} "
            f"dtype={data[key].dtype}"
        )

    if "action_mask" in data.files:
        source_action_mask = data["action_mask"]
    else:
        source_action_mask = np.ones(
            (
                len(data["proposed_action_id"]),
                len(ACTION_ID_TO_NAME),
            ),
            dtype=np.bool_,
        )

    print("\n===== Rollout Steps =====")

    total_steps = len(data["frame_id_end"])

    print(f"total steps: {total_steps}\n")

    header = (
        f"{'idx':>4} "
        f"{'frame':>6} "
        f"{'action':>7} "
        f"{'vis':>3} "
        f"{'phase':>8} "
        f"{'high':>6} "
        f"{'medium':>7} "
        f"{'low':>6} "
        f"{'a1_s':>5} "
        f"{'a1_e':>5} "
        f"{'a2_s':>5} "
        f"{'a2_e':>5} "
        f"{'p_hit':>5} "
        f"{'b_hit':>5} "
        f"{'done':>4}"
    )

    print(header)
    print("-" * len(header))

    for i in range(total_steps):
        resolved = resolve_priority_reward(
            reward_high=float(data["reward_high"][i]),
            reward_medium=float(data["reward_medium"][i]),
            reward_low=float(data["reward_low"][i]),
            ue_player_hit_count=int(
                data["ue_player_hit_count"][i]
            ),
            ue_boss_hit_count=int(
                data["ue_boss_hit_count"][i]
            ),
        )
        priority = PRIORITY_NAMES[resolved["priority"]]
        selected_reward = resolved["reward"]

        print(
            f"{i:>4} "
            f"{int(data['frame_id_end'][i]):>6} "
            f"{int(data['final_action_id'][i]):>7} "
            f"{int(data['visible'][i]):>3} "
            f"{str(data['phase'][i]):>8} "
            f"{float(data['reward_high'][i]):>6.2f} "
            f"{float(data['reward_medium'][i]):>7.2f} "
            f"{float(data['reward_low'][i]):>6.2f} "
            f"{int(data['ue_att1_start'][i]):>5} "
            f"{int(data['ue_att1_end'][i]):>5} "
            f"{int(data['ue_att2_start'][i]):>5} "
            f"{int(data['ue_att2_end'][i]):>5} "
            f"{int(data['ue_player_hit_count'][i]):>5} "
            f"{int(data['ue_boss_hit_count'][i]):>5} "
            f"{int(data['done'][i]):>4}"
            f"{priority:>4} "
            f"{selected_reward:>8.2f}"
        )

    print("\n===== Reward Consistency Check =====")

    mismatch_count = 0

    for i in range(total_steps):
        expected_medium = (
            int(data["ue_player_hit_count"][i])
            - int(data["ue_boss_hit_count"][i])
        )

        actual_medium = float(
            data["reward_medium"][i]
        )

        if not np.isclose(
            actual_medium,
            expected_medium,
        ):
            mismatch_count += 1

            print(
                f"[MISMATCH] "
                f"step={i} "
                f"frame={data['frame_id_end'][i]} "
                f"expected={expected_medium:.2f} "
                f"actual={actual_medium:.2f}"
            )

    if mismatch_count == 0:
        print("medium reward: OK")
    else:
        print(
            f"medium reward mismatches: "
            f"{mismatch_count}"
        )


if __name__ == "__main__":
    main()