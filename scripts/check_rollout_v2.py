import numpy as np
from pathlib import Path

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
    
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
        f"{'atk_s':>5} "
        f"{'atk_e':>5} "
        f"{'p_hit':>5} "
        f"{'b_hit':>5} "
        f"{'done':>4}"
    )

    print(header)
    print("-" * len(header))

    for i in range(total_steps):
        print(
            f"{i:>4} "
            f"{int(data['frame_id_end'][i]):>6} "
            f"{int(data['final_action_id'][i]):>7} "
            f"{int(data['visible'][i]):>3} "
            f"{str(data['phase'][i]):>8} "
            f"{float(data['reward_high'][i]):>6.2f} "
            f"{float(data['reward_medium'][i]):>7.2f} "
            f"{float(data['reward_low'][i]):>6.2f} "
            f"{int(data['ue_attack_start'][i]):>5} "
            f"{int(data['ue_attack_end'][i]):>5} "
            f"{int(data['ue_player_hit_count'][i]):>5} "
            f"{int(data['ue_boss_hit_count'][i]):>5} "
            f"{int(data['done'][i]):>4}"
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