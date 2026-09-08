import torch
import numpy as np
from pathlib import Path

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2/processed")

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

    print(data["actor_id"])
    print(data["actor_policy_step"].shape)
    print(data["actor_policy_step"])

if __name__ == "__main__":
    main()