import torch, torch.nn as nn
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet, TeacherPolicyNet)
from impala_unroll import build_unrolls
from impala_learner import warmstart_actor_critic_from_bc, set_impala_train_mode

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
BATCH_UNROLLS = 1

BC_WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")

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

    batch_size = frames.shape[0]
    unroll_length = frames.shape[1]

    flat_frames = frames.reshape(batch_size * unroll_length,*frames.shape[2:],)
    flat_extra = extra.reshape(batch_size * unroll_length,-1)

    flat_frames = torch.from_numpy(flat_frames).float()
    flat_extra = torch.from_numpy(flat_extra).float()

    bc_state = torch.load(BC_WEIGHTS_PATH,map_location="cpu")
    bc_model.load_state_dict(bc_state)

    warmstart_actor_critic_from_bc(actor_critic, bc_model)

    bc_logits = bc_model(flat_frames, flat_extra)
    ac_logits, _ = actor_critic(flat_frames, flat_extra)

    assert torch.allclose(bc_logits, ac_logits, atol=1e-6)

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

    set_impala_train_mode(actor_critic)

    assert actor_critic.training
    for bn in bn_modules:
        assert not bn.training

    running_means_before = [
        bn.running_mean.clone() for bn in bn_modules
    ]

    running_vars_before = [
        bn.running_var.clone() for bn in bn_modules
    ]

    actor_critic(flat_frames, flat_extra)

    for bn, mean_before, var_before in zip(bn_modules, running_means_before, running_vars_before):
        assert torch.equal(
            bn.running_mean,
            mean_before,
        )

        assert torch.equal(
            bn.running_var,
            var_before,
        )

    print("BatchNorm running stats unchanged: OK")

    for bn in bn_modules:
        assert bn.weight.requires_grad
        assert bn.bias.requires_grad

if __name__ == "__main__":
    main()