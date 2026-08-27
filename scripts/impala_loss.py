import torch

def compute_impala_loss(target_logits, target_action_log_prob, values, vs, pg_advantages, valid_mask, value_coef=0.5, entropy_coef=0.01):

    valid_count = valid_mask.sum().clamp_min(1.0)

    policy_terms = (target_action_log_prob * pg_advantages)
    policy_loss = -(policy_terms * valid_mask).sum() / valid_count

    value_errors = (vs - values) ** 2
    value_loss = (value_errors * valid_mask).sum() / valid_count

    log_probs = torch.log_softmax(target_logits, dim=-1)
    probs = torch.softmax(target_logits, dim=-1)
    entropy_terms = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
    entropy_per_step = -entropy_terms.sum(dim=-1)
    entropy = (entropy_per_step * valid_mask).sum() / valid_count

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy


    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }