import torch, torch.nn as nn
import numpy as np
from pathlib import Path
import time

from models import (TeacherActorCriticNet, TeacherPolicyNet)
from impala_unroll import build_unrolls
from impala_learner import train_impala_batch, warmstart_actor_critic_from_bc, set_impala_train_mode

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

BC_WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
SAVE_PATH = Path("data/meta/impala_persistent_smoke.pt")
UNROLL_LENGTH = 20

LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 40.0

actor_critic = TeacherActorCriticNet(
    in_ch=3,
    extra_dim=24,
    num_actions=10,
)

def main() -> None:
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

    while True:
        processed_any = False
        files = sorted(
                ROLLOUT_DIR.glob("*.npz"),
                key=lambda path: path.stat().st_mtime,
            )
        
        if not files:
            print(f"No rollout files found in {ROLLOUT_DIR}")
            time.sleep(5)
            continue
         
        for path in files:
            print(f"loading: {path}")

            with np.load(path, allow_pickle=False) as data:
                if "rollout_profile" not in data.files:
                    continue

                if not data["rollout_profile"] == "train":
                    continue

                if "rollout_profile" in data.files and data["rollout_profile"] == "train":
                    processed_any = True

                unrolls = build_unrolls(
                    data,
                    unroll_length=UNROLL_LENGTH,
                )

                if not unrolls:
                    print(f"No valid unrolls found in {path}")
                    skipped_dir = ROLLOUT_DIR / "skipped"
                    skipped_dir.mkdir(exist_ok=True)

                    skipped_path = skipped_dir / path.name
                    path.rename(skipped_path)
                    print(f"Moved {path} to {skipped_path}")
                    continue

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

            print(f"Finished processing: {path}")
            #move the processed file to another location
            processed_dir = ROLLOUT_DIR / "processed"
            processed_dir.mkdir(exist_ok=True)

            processed_path = processed_dir / path.name
            path.rename(processed_path)

        if not processed_any:
            print("No valid rollout files found.")
            time.sleep(5)

if __name__ == "__main__":
    main()