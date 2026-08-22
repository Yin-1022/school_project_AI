import numpy as np
from pathlib import Path

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

def build_full_unrolls(data, unroll_length=20):
    unrolls = []
    total_steps = len(data["proposed_action_id"])
    start = 0

    while start + unroll_length < total_steps:
        end = start + unroll_length

        chunk_done = data["done"][start:end]
        if np.any(chunk_done):
            start = end
            continue

        unroll_dict = {
            "frames": data["frames"][start:end],
            "extra": data["extra"][start:end],
            "proposed_action_id": data["proposed_action_id"][start:end],
            "final_action_id": data["final_action_id"][start:end],
            "reward_high": data["reward_high"][start:end],
            "reward_medium": data["reward_medium"][start:end],
            "reward_low": data["reward_low"][start:end],
            "done": data["done"][start:end],
            "probs": data["probs"][start:end],
            "behavior_probs": data["behavior_probs"][start:end],
            "bootstrap_frames": data["frames"][end],
            "bootstrap_extra": data["extra"][end],
        }
        unrolls.append(unroll_dict)
        start = end

    return unrolls

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
    unrolls = build_full_unrolls(data, unroll_length=20)
    print(len(unrolls))

    for i, unroll in enumerate(unrolls):
        print(f"===== Unroll {i} =====")
        print(
            "frames:", unroll["frames"].shape,
            "\nproposed_action_id:", unroll["proposed_action_id"].shape,
            "\nbootstrap_frames:", unroll["bootstrap_frames"].shape,
        )

if __name__ == "__main__":
    main()