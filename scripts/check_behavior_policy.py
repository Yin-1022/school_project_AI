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

    np.allclose(
        data["behavior_probs"].sum(axis=1),
        1.0,
        atol=1e-5,
    )
    total_steps = len(data["frame_id_end"])

    for i in range(total_steps):
        final_id = int(data["final_action_id"][i])
        actual_behavior_prob = (data["behavior_probs"][i, final_id])

    print(f"===== Behavior Policy Examples ====="
          f"frame={data['frame_id_end'][i]} "
          f"proposed={data['proposed_action_id'][i]} "
          f"final={data['final_action_id'][i]} "

          f"raw:"
          f"P({data['proposed_action_id'][i]})={data['behavior_probs'][i, data['proposed_action_id'][i]]:.4f} "
          f"P({data['final_action_id'][i]})={data['behavior_probs'][i, data['final_action_id'][i]]:.4f} "

          f"behavior:"
          f"μ({data['final_action_id'][i]}={actual_behavior_prob:.4f})")