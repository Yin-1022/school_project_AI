import torch, torch.nn as nn
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet, TeacherPolicyNet)
from impala_unroll import build_unrolls
from impala_learner import train_impala_batch, warmstart_actor_critic_from_bc, set_impala_train_mode

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

BC_WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
SAVE_PATH = Path("data/meta/impala_single_process.pt")
UNROLL_LENGTH = 20

LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 40.0

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

    bc_model = TeacherPolicyNet(
        in_ch=3,
        extra_dim=24,
        num_actions=10,
    )

    bc_state = torch.load(
        BC_WEIGHTS_PATH,
        map_location="cpu",
    )

    bc_model.load_state_dict(bc_state)

    warmstart_actor_critic_from_bc(
        actor_critic,
        bc_model,
    )

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    global_step = 0

    for path in files:
        print(f"loading: {path}")

        data = np.load(
            path,
            allow_pickle=False,
        )

        unrolls = build_unrolls(
            data,
            unroll_length=UNROLL_LENGTH,
        )

        for unroll in unrolls:
            metrics = train_impala_batch(
                model=actor_critic,
                optimizer=optimizer,
                batch_unrolls=[unroll],
                max_grad_norm=MAX_GRAD_NORM,
            )
            global_step += 1

            print(f"Step: {global_step}")
            print(f"Loss: {metrics['total_loss'].item()}")
            print(f"Policy: {metrics['policy_loss'].item()}")
            print(f"Value: {metrics['value_loss'].item()}")
            print(f"Entropy: {metrics['entropy'].item()}")
            print(f"mean_rho: {metrics['mean_rho'].item()}")
            print(f"grad_norm: {metrics['grad_norm'].item()}")
            print(f"Valid: {metrics['valid_steps'].item()}")

    torch.save(
        {
            "model_state_dict":
                actor_critic.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "training_step":
                global_step,
        },
        SAVE_PATH,
    )

if __name__ == "__main__":
    main()