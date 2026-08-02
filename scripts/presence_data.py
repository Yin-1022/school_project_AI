from pathlib import Path
import time

import numpy as np

PRESENCE_ROOT = Path("data/presence_samples/train/normal")

def save_presence_sample(
    frames: np.ndarray,
    present: bool,
    frame_id_end: int,
) -> Path:
    if frames.shape != (3, 8, 192, 192):
        raise ValueError(
            f"Expected frames shape (3, 8, 192, 192), got {frames.shape}"
        )

    label = 1 if present else 0
    class_name = "present" if present else "absent"

    output_dir = PRESENCE_ROOT / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.time_ns()
    output_path = output_dir / (
        f"presence_{timestamp}_{frame_id_end}.npz"
    )

    np.savez_compressed(
        output_path,
        frames=frames.astype(np.float32),
        label=np.int64(label),
        frame_id_end=np.int64(frame_id_end),
    )

    print(
        f"[presence-save] label={class_name} "
        f"frame={frame_id_end} path={output_path}"
    )

    return output_path