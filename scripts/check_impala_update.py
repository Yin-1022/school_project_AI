import torch, torch.nn as nn
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet, TeacherPolicyNet)
from impala_unroll import build_unrolls
from vtrace import compute_importance_ratios, compute_vtrace_value_targets, compute_policy_gradient_advantages, prepare_vtrace_weights
from impala_learner import warmstart_actor_critic_from_bc, set_impala_train_mode
from impala_loss import compute_impala_loss

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
BATCH_UNROLLS = 1

BC_WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
LEARNING_RATE = 1e-4

bc_model = TeacherPolicyNet(
    in_ch=3,
    extra_dim=24,
    num_actions=10,
)

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

    bootstrap_frames = np.stack([u["bootstrap_frames"] for u in batch_unrolls], axis=0)
    bootstrap_extra = np.stack([u["bootstrap_extra"] for u in batch_unrolls], axis=0)
    bootstrap_valid = np.stack([u["bootstrap_valid"] for u in batch_unrolls], axis=0)
    bootstrap_frames_tensor = torch.from_numpy(bootstrap_frames).float()
    bootstrap_extra_tensor = torch.from_numpy(bootstrap_extra).float()
    bootstrap_valid_tensor = torch.from_numpy(bootstrap_valid).float()

    selected_reward = np.stack([u["selected_reward"] for u in batch_unrolls])
    reward_tensor = torch.from_numpy(selected_reward).float()

    bc_state = torch.load(BC_WEIGHTS_PATH,map_location="cpu")
    bc_model.load_state_dict(bc_state)

    warmstart_actor_critic_from_bc(actor_critic, bc_model)
    set_impala_train_mode(actor_critic)

    with torch.no_grad():
        _, raw_bootstrap_value = actor_critic(bootstrap_frames_tensor,bootstrap_extra_tensor,)

    bootstrap_value = (raw_bootstrap_value* bootstrap_valid_tensor)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    bn_modules = [
            module
            for module in actor_critic.modules()
            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            )
        ]

    running_means_before = [
        bn.running_mean.clone() for bn in bn_modules
    ]

    running_vars_before = [
        bn.running_var.clone() for bn in bn_modules
    ]

    flat_logits, flat_values = actor_critic(flat_frames, flat_extra)

    logits = flat_logits.reshape(batch_size, unroll_length, -1)
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

    optimizer.zero_grad()
    losses["total_loss"].backward()

    grad_params = [
        p
        for p in actor_critic.parameters()
        if p.grad is not None
    ]
    assert len(grad_params) > 0
    for param in grad_params:
        assert torch.isfinite(param.grad).all()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        actor_critic.parameters(),
        max_norm=40.0,
    )

    policy_before = (
        actor_critic.policy_head.weight
        .detach()
        .clone()
    )

    value_before = (
        actor_critic.value_head.weight
        .detach()
        .clone()
    )

    trunk_before = (
        actor_critic.trunk[0].weight
        .detach()
        .clone()
    )

    optimizer.step()

    assert (actor_critic.policy_head.weight.grad is not None)
    assert (actor_critic.value_head.weight.grad is not None)
    assert (actor_critic.trunk[0].weight.grad is not None)

    assert not torch.equal(
        actor_critic.policy_head.weight,
        policy_before,
    )

    assert not torch.equal(
        actor_critic.value_head.weight,
        value_before,
    )

    assert not torch.equal(
        actor_critic.trunk[0].weight,
        trunk_before,
    )

    for bn, mean_before, var_before in zip(bn_modules, running_means_before, running_vars_before):
        assert torch.equal(
            bn.running_mean,
            mean_before,
        )

        assert torch.equal(
            bn.running_var,
            var_before,
        )

    print("===== IMPALA One-Step Update Check =====")

    print(f"policy_loss: {losses['policy_loss'].item():.6f}")
    print(f"value_loss: {losses['value_loss'].item():.6f}")
    print(f"entropy: {losses['entropy'].item():.6f}")
    print(f"total_loss: {losses['total_loss'].item():.6f}")

    print("Backward: OK")
    print("Policy head gradient: OK")
    print("Value head gradient: OK")
    print("Shared trunk gradient: OK")

    print(f"Gradient norm before clipping: {grad_norm:.6f}")
    print(f"Gradient finite check: OK")
    print(f"Gradient clipping: OK")

    print("Policy head updated: OK")
    print("Value head updated: OK")
    print("Shared trunk updated: OK")

    print("BatchNorm running stats unchanged: OK")

if __name__ == "__main__":
    main()