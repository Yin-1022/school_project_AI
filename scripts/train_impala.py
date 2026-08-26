import torch, torch.nn as nn
import numpy as np
from pathlib import Path

from models import (TeacherActorCriticNet)
from impala_unroll import build_unrolls
from impala_learner import train_impala_batch

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")
BATCH_UNROLLS = 1

BC_WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
SAVE_PATH = Path("data/meta/impala_single_process.pt")
UNROLL_LENGTH = 20

LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 40.0

NUM_EPOCHS = 1

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
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    global_step = 0
    for unroll in unrolls:
        Loss =train_impala_batch(
            actor_critic=actor_critic,
            optimizer=optimizer,
            unroll=[unroll]
        )
        global_step += 1

        print(f"Step: {global_step}")
        print(f"Loss: {Loss['total_loss'].item()}")
        print(f"Policy: {Loss['policy_loss'].item()}")
        print(f"Value: {Loss['value_loss'].item()}")
        print(f"Entropy: {Loss['entropy'].item()}")
        print(f"mean_rho: {Loss['mean_rho'].item()}")
        print(f"grad_norm: {Loss['grad_norm'].item()}")
        print(f"Valid: {Loss['valid_steps'].item()}")

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