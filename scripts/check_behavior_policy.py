import numpy as np
from pathlib import Path
from constant import ACTION_ID_TO_NAME

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

    behavior_sum_ok = np.allclose(
        data["behavior_probs"].sum(axis=1),
        1.0,
        atol=1e-5,
    )

    print(
        "Behavior sums to 1:",
        "OK" if behavior_sum_ok else "FAILED",
    )
    total_steps = len(data["frame_id_end"])

    for i in range(total_steps):
        proposed_id = int(data["proposed_action_id"][i])
        final_id = int(data["final_action_id"][i])
        raw_proposed_prob = (data["behavior_probs"][i, proposed_id])
        raw_final_prob = (data["behavior_probs"][i, final_id])
        actual_behavior_prob = (data["behavior_probs"][i, final_id])

    if actual_behavior_prob <= 0:
        raise ValueError(
            f"Final action has zero behavior probability "
            f"at step {i}"
        )

    proposed_action_name = ACTION_ID_TO_NAME[proposed_id]
    final_action_name = ACTION_ID_TO_NAME[final_id]
    print(f"===== Behavior Policy Examples =====\n"
          f"frame={data['frame_id_end'][i]} \n"
          f"proposed={proposed_action_name} \n"
          f"final={final_action_name} \n"

          f"raw:\n"
          f" P({proposed_action_name})={raw_proposed_prob:.4f} \n"
          f" P({final_action_name})={raw_final_prob:.4f} \n"

          f"behavior:\n"
          f" μ({final_action_name})={actual_behavior_prob:.4f}\n")

if __name__ == "__main__":
    main()