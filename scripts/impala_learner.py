import torch, torch.nn as nn


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
        if isinstance(module, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
            module.eval()