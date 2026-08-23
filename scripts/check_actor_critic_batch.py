import torch
import numpy as np
from pathlib import Path

from models import (
    TeacherPolicyNet,
    TeacherActorCriticNet,
)

from impala_unroll import build_unrolls

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
BATCH_UNROLLS = 1

actor_critic = TeacherActorCriticNet(
    in_ch=3,
    extra_dim=24,
    num_actions=10,
)

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
    batch_unrolls = unrolls[:BATCH_UNROLLS]

    frames = np.stack(
        [u["frames"] for u in batch_unrolls],
        axis=0,
    )

    extra = np.stack(
        [u["extra"] for u in batch_unrolls],
        axis=0,
    )

    batch_size = frames.shape[0]
    unroll_length = frames.shape[1]

    flat_frames = frames.reshape(batch_size * unroll_length,*frames.shape[2:],)
    flat_extra = extra.reshape(batch_size * unroll_length,-1)

    flat_frames = torch.from_numpy(flat_frames).float()
    flat_extra = torch.from_numpy(flat_extra).float()

    with torch.no_grad():
        flat_logits, flat_values = actor_critic(flat_frames, flat_extra)

    logits = flat_logits.reshape(
        batch_size,
        unroll_length,
        -1,
    )

    values = flat_values.reshape(
        batch_size,
        unroll_length,
    )

    proposed_action_id = np.stack(
        [u["proposed_action_id"] for u in batch_unrolls],
    )

    selected_reward = np.stack(
        [u["selected_reward"] for u in batch_unrolls],
    )

    behavior_log_prob = np.stack(
        [u["behavior_log_prob"] for u in batch_unrolls],
    )

    done = np.stack(
        [u["done"] for u in batch_unrolls],
    )

    valid_mask = np.stack(
        [u["valid_mask"] for u in batch_unrolls],
    )

    assert logits.shape == (
        BATCH_UNROLLS,
        20,
        10,
    )

    assert values.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert proposed_action_id.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert selected_reward.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert behavior_log_prob.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert valid_mask.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    print("Actor-Critic batch smoke test: OK")