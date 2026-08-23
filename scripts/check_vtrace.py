import torch
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet,)
from impala_unroll import build_unrolls
from vtrace import compute_importance_ratios

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

    action = np.stack([u["proposed_action_id"] for u in batch_unrolls])
    action_tensor = torch.from_numpy(action).long()

    behavior_log_prob = np.stack([u["behavior_log_prob"] for u in batch_unrolls])
    behavior_log_prob_tensor = torch.from_numpy(behavior_log_prob).float()
    
    batch_size = frames.shape[0]
    unroll_length = frames.shape[1]

    flat_frames = frames.reshape(batch_size * unroll_length,*frames.shape[2:],)
    flat_extra = extra.reshape(batch_size * unroll_length,-1)

    flat_frames = torch.from_numpy(flat_frames).float()
    flat_extra = torch.from_numpy(flat_extra).float()

    with torch.no_grad():
        flat_logits, _ = actor_critic(flat_frames, flat_extra)

    logits = flat_logits.reshape(
        batch_size,
        unroll_length,
        -1,
    )

    target_action_log_prob, log_rhos, rhos = (
        compute_importance_ratios(
            target_logits=logits,
            actions=action_tensor,
            behavior_log_prob=behavior_log_prob_tensor,
        )
    )

    assert target_action_log_prob.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert log_rhos.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert rhos.shape == (
        BATCH_UNROLLS,
        20,
    )

    assert torch.isfinite(
        target_action_log_prob
    ).all()

    assert torch.isfinite(
        log_rhos
    ).all()

    assert torch.isfinite(
        rhos
    ).all()

    print("===== V-trace Importance Ratio Check =====\n")
    print(f"target_action_log_prob: {target_action_log_prob.shape}")
    print(f"log_rhos: {log_rhos.shape}")
    print(f"rhos: {rhos.shape}\n")
    print("Valid rho stats:")
    print(f"min: {rhos.min().item()}")
    print(f"max: {rhos.max().item()}")
    print(f"mean: {rhos.mean().item()}")
    print(f"median: {rhos.median().item()}")

    print("Importance ratio calculation: OK")

    b = 0
    t = 0
    action_id = action[b, t]
    manual_log_pi = torch.log_softmax(
        logits[b, t],
        dim=-1,
    )[action_id]

    manual_log_mu = behavior_log_prob[b, t]

    manual_rho = torch.exp(
        manual_log_pi
        - manual_log_mu
    )

    assert torch.allclose(
        rhos[b, t],
        manual_rho,
        atol=1e-6,
    )

    print("V-trace manual check: OK")

if __name__ == "__main__":
    main()