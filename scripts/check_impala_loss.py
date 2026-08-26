import torch
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet,)
from impala_unroll import build_unrolls
from vtrace import compute_importance_ratios, prepare_vtrace_weights, compute_vtrace_value_targets, compute_policy_gradient_advantages
from impala_loss import compute_impala_loss

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
    terminal_unroll = next(u for u in unrolls if u["bootstrap_valid"] == 0)
    batch_unrolls = [terminal_unroll]

    frames = np.stack(
        [u["frames"] for u in batch_unrolls],
        axis=0,
    )

    extra = np.stack(
        [u["extra"] for u in batch_unrolls],
        axis=0,
    )

    done = np.stack(
        [u["done"] for u in batch_unrolls],
        axis=0,
    )

    valid_mask = np.stack(
        [u["valid_mask"] for u in batch_unrolls],
        axis=0,
    )

    action = np.stack([u["proposed_action_id"] for u in batch_unrolls])
    action_tensor = torch.from_numpy(action).long()

    behavior_log_prob = np.stack([u["behavior_log_prob"] for u in batch_unrolls])
    behavior_log_prob_tensor = torch.from_numpy(behavior_log_prob).float()

    done_tensor = torch.from_numpy(done).float()
    valid_mask_tensor = torch.from_numpy(valid_mask).float()
    
    batch_size = frames.shape[0]
    unroll_length = frames.shape[1]

    flat_frames = frames.reshape(batch_size * unroll_length,*frames.shape[2:],)
    flat_extra = extra.reshape(batch_size * unroll_length,-1)

    flat_frames = torch.from_numpy(flat_frames).float()
    flat_extra = torch.from_numpy(flat_extra).float()

    with torch.no_grad():
        flat_logits, flat_values = actor_critic(flat_frames, flat_extra)

    logits = flat_logits.reshape(batch_size, unroll_length, -1,)
    values = flat_values.reshape(batch_size, unroll_length,)

    target_action_log_prob, _, rhos = (
        compute_importance_ratios(
            target_logits=logits,
            actions=action_tensor,
            behavior_log_prob=behavior_log_prob_tensor,
        )
    )

    clipped_rhos, cs, discounts = prepare_vtrace_weights(
        rhos=rhos,
        done=done_tensor,
        valid_mask=valid_mask_tensor,
    )

    terminal_unroll = next(u for u in unrolls if u["bootstrap_valid"] == 0)

    selected_reward = np.stack([u["selected_reward"] for u in batch_unrolls])
    reward_tensor = torch.from_numpy(selected_reward).float()

    bootstrap_frames = np.stack([u["bootstrap_frames"] for u in batch_unrolls], axis=0)
    bootstrap_extra = np.stack([u["bootstrap_extra"] for u in batch_unrolls], axis=0)
    bootstrap_valid = np.stack([u["bootstrap_valid"] for u in batch_unrolls], axis=0)
    bootstrap_frames_tensor = torch.from_numpy(bootstrap_frames).float()
    bootstrap_extra_tensor = torch.from_numpy(bootstrap_extra).float()
    bootstrap_valid_tensor = torch.from_numpy(bootstrap_valid).float()

    with torch.no_grad():
        _, raw_bootstrap_value = actor_critic(bootstrap_frames_tensor,bootstrap_extra_tensor,)

    bootstrap_value = (raw_bootstrap_value* bootstrap_valid_tensor)

    vs = compute_vtrace_value_targets(
        rewards=reward_tensor,
        values=values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        clipped_rhos=clipped_rhos,
        cs=cs,
        valid_mask=valid_mask_tensor,
    )

    pg_advantages = compute_policy_gradient_advantages(
        rewards=reward_tensor,
        values=values,
        vs=vs,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        clipped_rhos=clipped_rhos,
        valid_mask=valid_mask_tensor,
    )

    losses = compute_impala_loss(
        target_logits=logits,
        target_action_log_prob=target_action_log_prob,
        values=values,
        vs=vs,
        pg_advantages=pg_advantages,
        valid_mask=valid_mask_tensor,
    )

    assert losses["total_loss"].ndim == 0
    assert losses["policy_loss"].ndim == 0
    assert losses["value_loss"].ndim == 0
    assert losses["entropy"].ndim == 0
    assert torch.isfinite(losses["total_loss"])

    print("===== IMPALA Loss Check =====")
    print(f"valid steps: {int(valid_mask_tensor.sum().item())}\n")
    print(f"policy_loss: {losses['policy_loss'].item():.6f}")
    print(f"value_loss: {losses['value_loss'].item():.6f}")
    print(f"entropy: {losses['entropy'].item():.6f}")
    print(f"total_loss: {losses['total_loss'].item():.6f}")

    print("===== IMPALA loss calculation: OK =====")

if __name__ == "__main__":
    main()