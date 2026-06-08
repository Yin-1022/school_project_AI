from pathlib import Path
import numpy as np
from collections import Counter

sample_dir = Path("data/teacher_samples")
counter = Counter()

for npz_path in sorted(sample_dir.glob("*.npz")):
    data = np.load(npz_path, allow_pickle=True)
    action_name = str(data["action_name"])
    counter[action_name] += 1

print("Action distribution:")
for k, v in counter.items():
    print(f"{k}: {v}")