import torch
import numpy as np
from pathlib import Path

from impala_unroll import build_unrolls
from models import TeacherActorCriticNet

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

model = TeacherActorCriticNet(
    in_ch=3,
    extra_dim=24,
    num_actions=10,
)

model.eval()  # Set the model to evaluation mode

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
    unroll = unrolls[0]

    frames = torch.from_numpy(unroll["frames"]).float()
    extra = torch.from_numpy(unroll["extra"]).float()
    bootstrap_frames = torch.from_numpy(unroll["bootstrap_frames"]).unsqueeze(0).float()
    bootstrap_extra = torch.from_numpy(unroll["bootstrap_extra"]).unsqueeze(0).float()

    with torch.no_grad():
        logits, values = model(frames, extra)
        bootstrap_logits, bootstrap_values = model(bootstrap_frames, bootstrap_extra)

    assert logits.shape == (20, 10)
    assert values.shape == (20, 1)
    assert bootstrap_logits.shape == (1, 10)
    assert bootstrap_values.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(bootstrap_logits).all()
    assert torch.isfinite(bootstrap_values).all()