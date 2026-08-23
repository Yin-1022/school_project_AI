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

actor_critic.eval()

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
    terminal_unroll = unrolls[-1]

    frames = np.stack(
        [u["frames"] for u in batch_unrolls],
        axis=0,
    )

    extra = np.stack(
        [u["extra"] for u in batch_unrolls],
        axis=0,
    )

    bootstrap_frames = np.stack(
        [u["bootstrap_frames"] for u in batch_unrolls],
        axis=0,
    )
    bootstrap_extra = np.stack(
        [u["bootstrap_extra"] for u in batch_unrolls],
        axis=0,
    )
    bootstrap_valid = np.asarray(
        [u["bootstrap_valid"] for u in batch_unrolls],
        dtype=np.float32,
    )

    batch_size = frames.shape[0]
    unroll_length = frames.shape[1]

    flat_frames = frames.reshape(batch_size * unroll_length,*frames.shape[2:],)
    flat_extra = extra.reshape(batch_size * unroll_length,-1)

    flat_frames = torch.from_numpy(flat_frames).float()
    flat_extra = torch.from_numpy(flat_extra).float()
    bootstrap_frames_tensor = torch.from_numpy(bootstrap_frames).float()
    bootstrap_extra_tensor = torch.from_numpy(bootstrap_extra).float()
    bootstrap_valid_tensor = torch.from_numpy(bootstrap_valid).float()
    terminal_bootstrap_frames = (
         torch.from_numpy(terminal_unroll["bootstrap_frames"]).unsqueeze(0).float()
    )
    terminal_bootstrap_extra = (
         torch.from_numpy(terminal_unroll["bootstrap_extra"]).unsqueeze(0).float()
    )

    with torch.no_grad():
        flat_logits, flat_values = actor_critic(flat_frames, flat_extra)

    with torch.no_grad():
        _, raw_bootstrap_values = actor_critic(
            bootstrap_frames_tensor,
            bootstrap_extra_tensor,
        )

    with torch.no_grad():
        _, terminal_raw_value = actor_critic(
            terminal_bootstrap_frames,
            terminal_bootstrap_extra,
        )
        
    bootstrap_values = (
        raw_bootstrap_values
        * bootstrap_valid_tensor
    )

    terminal_value = (
        terminal_raw_value
        * float(terminal_unroll["bootstrap_valid"])
    )

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

    assert done.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert bootstrap_frames.shape == (
        BATCH_UNROLLS,
        3,
        8,
        192,
        192,
    )

    assert bootstrap_extra.shape == (
        BATCH_UNROLLS,
        24,
    )

    assert bootstrap_valid.shape == (
        BATCH_UNROLLS,
    )

    assert raw_bootstrap_values.shape == (
        BATCH_UNROLLS,
    )

    assert bootstrap_values.shape == (
        BATCH_UNROLLS,
    )

    assert torch.allclose(
        terminal_value,
        torch.zeros_like(terminal_value),
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()

    print("===== Actor-Critic Batch Check =====")
    print(f"frames: {frames.shape}")
    print(f"logits: {logits.shape}")
    print(f"values: {values.shape}")
    print(f"bootstrap_frames: {bootstrap_frames.shape}")
    print(f"bootstrap_extra: {bootstrap_extra.shape}")
    print(f"raw_bootstrap_values: {raw_bootstrap_values.shape}")
    print(f"bootstrap_values: {bootstrap_values.shape}")

    print("Bootstrap batch forward: OK")
    print("Terminal bootstrap check: OK")
    print("Actor-Critic batch smoke test: OK")

if __name__ == "__main__":
    main()