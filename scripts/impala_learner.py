import torch, torch.nn as nn
import numpy as np

from vtrace import compute_importance_ratios, compute_vtrace_value_targets, compute_policy_gradient_advantages, prepare_vtrace_weights
from impala_loss import compute_impala_loss

def warmstart_actor_critic_from_bc(actor_critic, bc_model):
    ac_state = actor_critic.state_dict()
    bc_state = bc_model.state_dict()
    for key, value in bc_state.items():
        if key.startswith("visual.") or key.startswith("extra_mlp."):
            ac_state[key] = value

    ac_state["trunk.0.weight"] = bc_state["head.0.weight"]
    ac_state["trunk.0.bias"] = bc_state["head.0.bias"]
    ac_state["policy_head.weight"] = bc_state["head.2.weight"]
    ac_state["policy_head.bias"] = bc_state["head.2.bias"]

    actor_critic.load_state_dict(ac_state)

    bc_model.eval()
    actor_critic.eval()

def set_impala_train_mode(model):
    model.train()

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()

def train_impala_batch(model, optimizer, batch_unrolls, max_grad_norm=40.0):
    set_impala_train_mode(model)

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

    action_mask = np.stack(
        [
            u["action_mask"]
            for u in batch_unrolls
        ],
        axis=0,
    )

    batch_size = frames.shape[0]
    unroll_length = frames.shape[1]

    action = np.stack([u["proposed_action_id"] for u in batch_unrolls])
    action_tensor = torch.from_numpy(action).long()
    behavior_log_prob = np.stack([u["behavior_log_prob"] for u in batch_unrolls])
    behavior_log_prob_tensor = torch.from_numpy(behavior_log_prob).float()
    done_tensor = torch.from_numpy(done).float()
    valid_mask_tensor = torch.from_numpy(valid_mask).float()
    action_mask_tensor = torch.from_numpy(action_mask).bool()

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

    with torch.no_grad():
        _, raw_bootstrap_value = model(bootstrap_frames_tensor, bootstrap_extra_tensor)

    bootstrap_value = (raw_bootstrap_value * bootstrap_valid_tensor)

    selected_reward = np.stack([u["selected_reward"] for u in batch_unrolls])
    reward_tensor = torch.from_numpy(selected_reward).float()
    
    flat_logits, flat_values = model(flat_frames, flat_extra)

    logits = flat_logits.reshape(batch_size, unroll_length, -1)
    values = flat_values.reshape(batch_size, unroll_length,)

    masked_logits = logits.masked_fill(
        ~action_mask_tensor.unsqueeze(-1),
        float("-inf"),
    )

    target_action_log_prob, _, rhos = (
        compute_importance_ratios(
            target_logits=masked_logits,
            actions=action_tensor,
            behavior_log_prob=behavior_log_prob_tensor,
            action_mask=action_mask_tensor,
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
        target_logits=masked_logits,
        target_action_log_prob=target_action_log_prob,
        values=values,
        vs=vs,
        pg_advantages=pg_advantages,
        valid_mask=valid_mask_tensor,
    )

    if not torch.isfinite(losses["total_loss"]):
        raise FloatingPointError(
            f"Non-finite IMPALA loss: {losses['total_loss'].item()}"
        )

    optimizer.zero_grad(set_to_none=True)
    losses["total_loss"].backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )

    optimizer.step()

    valid_steps = valid_mask_tensor.sum()
    valid_rhos = rhos[valid_mask_tensor.bool()]
    mean_rho = valid_rhos.mean()

    return {
        "total_loss": losses["total_loss"].detach(),
        "policy_loss": losses["policy_loss"].detach(),
        "value_loss": losses["value_loss"].detach(),
        "entropy": losses["entropy"].detach(),
        "grad_norm": grad_norm.detach(),
        "mean_rho": mean_rho.detach(),
        "valid_steps": valid_steps.detach(),
    }