import numpy as np
from pathlib import Path

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

def pad_first_dim(array, target_length):
    current_length = len(array)

    if current_length >= target_length:
        return array[:target_length]

    padding_shape = target_length - current_length, *array.shape[1:]
    padding = np.zeros(padding_shape, dtype=array.dtype)
    return np.concatenate([array, padding], axis=0)

def build_unrolls(data, unroll_length=20):
    unrolls = []
    total_steps = len(data["proposed_action_id"])
    start = 0

    while start < total_steps:
        remaining_steps = total_steps - start

        chunk_done = data["done"][start:start + unroll_length]
        if np.any(chunk_done):
            first_done_index = np.argmax(chunk_done)
            valid_length = first_done_index + 1
            valid_mask = np.zeros(unroll_length,dtype=np.float32)
            valid_mask[:valid_length] = 1.0
            bootstrap_valid = 0

            unroll_dict = {
                "frames": pad_first_dim(data["frames"][start:start + valid_length], unroll_length),
                "extra": pad_first_dim(data["extra"][start:start + valid_length], unroll_length),
                "proposed_action_id": pad_first_dim(data["proposed_action_id"][start:start + valid_length], unroll_length),
                "final_action_id": pad_first_dim(data["final_action_id"][start:start + valid_length], unroll_length),
                "reward_high": pad_first_dim(data["reward_high"][start:start + valid_length], unroll_length),
                "reward_medium": pad_first_dim(data["reward_medium"][start:start + valid_length], unroll_length),
                "reward_low": pad_first_dim(data["reward_low"][start:start + valid_length], unroll_length),
                "done": pad_first_dim(data["done"][start:start + valid_length], unroll_length),
                "probs": pad_first_dim(data["probs"][start:start + valid_length], unroll_length),
                "behavior_probs": pad_first_dim(data["behavior_probs"][start:start + valid_length], unroll_length),
                "bootstrap_frames": np.zeros_like(data["frames"][0]),
                "bootstrap_extra": np.zeros_like(data["extra"][0]),
                "valid_mask": pad_first_dim(valid_mask, unroll_length),
                "bootstrap_valid": np.int64(bootstrap_valid),
            }
            start = start + valid_length
        elif remaining_steps > unroll_length:
            valid_mask = np.ones(unroll_length,dtype=np.float32)
            bootstrap_valid = 1
            unroll_dict = {
                "frames": data["frames"][start:start + unroll_length],
                "extra": data["extra"][start:start + unroll_length],
                "proposed_action_id": data["proposed_action_id"][start:start + unroll_length],
                "final_action_id": data["final_action_id"][start:start + unroll_length],
                "reward_high": data["reward_high"][start:start + unroll_length],
                "reward_medium": data["reward_medium"][start:start + unroll_length],
                "reward_low": data["reward_low"][start:start + unroll_length],
                "done": data["done"][start:start + unroll_length],
                "probs": data["probs"][start:start + unroll_length],
                "behavior_probs": data["behavior_probs"][start:start + unroll_length],
                "bootstrap_frames": data["frames"][start + unroll_length],
                "bootstrap_extra": data["extra"][start + unroll_length],
                "valid_mask": valid_mask,
                "bootstrap_valid": np.int64(bootstrap_valid),
            }
            start = start + unroll_length
        else:
            bootstrap_valid = 0
            break

        unrolls.append(unroll_dict)

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
    unrolls = build_unrolls(data, unroll_length=20)
    print(len(unrolls))

    for i, unroll in enumerate(unrolls):
        terminal = np.any(unroll["done"][unroll["valid_mask"] == 1])
        print(f"===== Unroll {i} =====")
        print(
            "frames:", unroll["frames"].shape,
            "\nproposed_action_id:", unroll["proposed_action_id"].shape,
            "\nbootstrap_frames:", unroll["bootstrap_frames"].shape,
            "\nbootstrap_valid:", unroll["bootstrap_valid"],
            "\nvalid_steps:", unroll["valid_mask"].sum(),
            "\nterminal:", terminal,
        )

if __name__ == "__main__":
    main()