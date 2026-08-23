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