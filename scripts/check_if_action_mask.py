import numpy as np
from pathlib import Path

files = sorted(
    Path("data/rollouts/rollouts_bc_v2").glob("*.npz"),
    key=lambda p: p.stat().st_mtime,
)

path = files[-1]

with np.load(path, allow_pickle=False) as data:
    print(path)
    print(data.files)

    if "action_mask" in data.files:
        print("action_mask:", data["action_mask"].shape)