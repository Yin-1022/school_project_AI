import torch
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet,)
from impala_unroll import build_unrolls
from vtrace import compute_importance_ratios, prepare_vtrace_weights, compute_vtrace_value_targets

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
    flat_values = torch.zeros((batch_size * unroll_length, 1))

    with torch.no_grad():
        flat_logits, flat_values = actor_critic(flat_frames, flat_extra)

    logits = flat_logits.reshape(batch_size, unroll_length, -1,)
    values = flat_values.reshape(batch_size, unroll_length,)

    target_action_log_prob, log_rhos, rhos = (
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

    assert target_action_log_prob.shape == (BATCH_UNROLLS,20,)

    assert log_rhos.shape == (BATCH_UNROLLS,20,)

    assert rhos.shape == (BATCH_UNROLLS,20,)

    assert clipped_rhos.shape == rhos.shape

    assert cs.shape == rhos.shape

    assert discounts.shape == rhos.shape

    assert torch.isfinite(target_action_log_prob).all()

    assert torch.isfinite(log_rhos).all()

    assert torch.isfinite(rhos).all()

    assert torch.all(clipped_rhos <= 1.0 + 1e-6)

    assert torch.all(cs <= 1.0 + 1e-6)

    terminal_unroll = next(u for u in unrolls if u["bootstrap_valid"] == 0)
    terminal_done = torch.from_numpy(terminal_unroll["done"]).unsqueeze(0).float()
    terminal_valid_mask = torch.from_numpy(terminal_unroll["valid_mask"]).unsqueeze(0).float()
    terminal_rhos = torch.ones_like(terminal_done)

    clipped_rhos, cs, terminal_discounts = (
        prepare_vtrace_weights(
            rhos=terminal_rhos,
            done=terminal_done,
            valid_mask=terminal_valid_mask,
            gamma=0.99,
        )
    )

    valid_steps = int(terminal_valid_mask[0].sum().item())
    terminal_index = valid_steps - 1

    assert terminal_done[0, terminal_index] == 1

    assert terminal_discounts[0, terminal_index] == 0

    assert torch.all(terminal_discounts[0, valid_steps:] == 0)

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

    assert vs.shape == values.shape
    assert torch.isfinite(vs).all()
    assert torch.all(vs[valid_mask_tensor == 0] == 0)

    t = valid_steps - 1
    expected_terminal_v = (
        values[:, t]
        + clipped_rhos[:, t]
        * (
            reward_tensor[:, t]
            - values[:, t]
        )
    )

    assert torch.allclose(
        vs[0, t],
        expected_terminal_v,
        atol=1e-6,
    )

    valid_rhos = rhos[valid_mask_tensor.bool()]

    print("===== V-trace Importance Ratio Check =====\n")
    print(f"target_action_log_prob: {target_action_log_prob.shape}")
    print(f"log_rhos: {log_rhos.shape}")
    print(f"rhos: {rhos.shape}\n")
    print("Valid rho stats:")
    print(f"min: {valid_rhos.min().item()}")
    print(f"max: {valid_rhos.max().item()}")
    print(f"mean: {valid_rhos.mean().item()}")
    print(f"median: {valid_rhos.median().item()}")
    print(f"terminal done: {terminal_done[0]}")
    print(f"terminal valid mask: {terminal_valid_mask[0]}")
    print(f"terminal discounts: {terminal_discounts[0]}")

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