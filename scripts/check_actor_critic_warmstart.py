import torch
import numpy as np
from pathlib import Path

from models import (
    TeacherPolicyNet,
    TeacherActorCriticNet,
)

from impala_unroll import build_unrolls
from models import TeacherActorCriticNet

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

BC_WEIGHTS_PATH = "data/meta/best_teacher_policy.pt"

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

bc_state = torch.load(
    BC_WEIGHTS_PATH,
    map_location="cpu",
)

bc_model.load_state_dict(bc_state)

ac_state = actor_critic.state_dict()
for key, value in bc_state.items():
    if key.startswith("visual.") or key.startswith("extra_mlp."):
        ac_state[key] = value

ac_state["trunk.0.weight"] = bc_state["head.0.weight"]
ac_state["trunk.0.bias"] = bc_state["head.0.bias"]
ac_state["policy_head.weight"] = bc_state["head.2.weight"]
ac_state["policy_head.bias"] = bc_state["head.2.bias"]

actor_critic.load_state_dict(ac_state)

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
    unroll = unrolls[0]

    frames = torch.from_numpy(unroll["frames"]).float()
    extra = torch.from_numpy(unroll["extra"]).float()

    with torch.no_grad():
        bc_logits = bc_model(
            frames,
            extra,
        )

        ac_logits, ac_values = actor_critic(
            frames,
            extra,
        )

    assert torch.allclose(
        bc_logits,
        ac_logits,
        atol=1e-6,
    )
    assert torch.isfinite(
        ac_values
    ).all()

    bc_action = bc_logits.argmax(dim=1)
    ac_action = ac_logits.argmax(dim=1)

    assert torch.equal(
        bc_action,
        ac_action,
    )

    print("Actor-Critic warmstart check: OK")

if __name__ == "__main__":
    main()