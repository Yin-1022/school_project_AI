import torch

def compute_importance_ratios(target_logits, actions, behavior_log_prob,):
    target_log_prob = torch.log_softmax(target_logits, dim=-1)
    target_action_log_prob = torch.gather(
            target_log_prob, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

    #Ratio between Actor收集資料時的policy & 當前Learner對此action之看法
    log_rhos = target_action_log_prob - behavior_log_prob
    rhos = torch.exp(log_rhos)

    return target_action_log_prob, log_rhos, rhos

def prepare_vtrace_weights(rhos, done, valid_mask, gamma=0.99, rho_clip=1.0, c_clip=1.0):
    #「這一步自己的 TD correction 要信多少」 
    clipped_rhos = torch.clamp(rhos, max=rho_clip)

    #「未來修正允許往前流多少」的閥門
    cs = torch.clamp(rhos, max=c_clip)

    done = done.float()
    valid_mask = valid_mask.float()

    discounts = (
        gamma * (1.0 - done) * valid_mask
    )

    clipped_rhos = clipped_rhos * valid_mask
    cs = cs * valid_mask

    return clipped_rhos, cs, discounts

@torch.no_grad()
def compute_vtrace_value_targets(rewards, values, bootstrap_value, discounts, clipped_rhos, cs, valid_mask):
    # V(s_{t+1})
    next_values = torch.cat([values[:, 1:], bootstrap_value.unsqueeze(1)], dim=1)

    # δ_t^V
    deltas = clipped_rhos * (rewards + discounts * next_values - values)

    # 儲存 v_t
    vs = torch.zeros_like(values)

    # 一開始持有 v_T = V(s_T)
    next_vtrace = bootstrap_value

    for t in reversed(range(values.shape[1])):
        # v_t =
        # V(s_t)
        # + δ_t^V
        # + γ_t c_t [v_{t+1} - V(s_{t+1})]
        current_vtrace = (
            values[:, t] 
            + deltas[:, t] 
            + discounts[:, t] * cs[:, t] * (next_vtrace - next_values[:, t])
        )

        is_valid = valid_mask[:, t].bool()
        next_vtrace = torch.where(is_valid, current_vtrace, next_vtrace)
        vs[:, t] = torch.where(is_valid, next_vtrace, torch.zeros_like(next_vtrace))

    return vs