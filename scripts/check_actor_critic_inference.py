import numpy as np
import torch

from policy_inference import (
    load_actor_critic_model,
    infer_actor_critic_action,
)

CHECKPOINT_PATH = "data/meta/impala_persistent_smoke.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = load_actor_critic_model(
        CHECKPOINT_PATH,
        device=device,
    )

    # 模擬一筆 observation
    frames = torch.rand(
        1, 3, 8, 192, 192,
        dtype=torch.float32,
    )

    extra = torch.zeros(
        1, 24,
        dtype=torch.float32,
    )

    # 10 個 action，先全部允許
    action_mask = np.ones(10, dtype=bool)

    # 模擬兩個 action 不合法
    # EvadeBack = 4
    # Retreat   = 5
    action_mask[4] = False
    action_mask[5] = False

    result = infer_actor_critic_action(
        frames=frames,
        extra=extra,
        model=model,
        sample=True,
        action_mask=action_mask,
    )

    action_id = result["action_id"]
    action_name = result["action_name"]
    logits = result["logits"]
    probs = result["probs"]
    value = result["value"]

    print()
    print("===== Actor-Critic Inference =====")
    print("action_id:", action_id)
    print("action_name:", action_name)
    print("logits shape:", tuple(logits.shape))
    print("probs shape:", tuple(probs.shape))
    print("value shape:", tuple(value.shape))
    print("prob sum:", probs.sum(dim=1).item())
    print("value:", value.item())

    print()
    print("Masked probabilities:")
    print("EvadeBack:", probs[0, 4].item())
    print("Retreat:", probs[0, 5].item())

    # -------- Assertions --------

    assert logits.shape == (1, 10), (
        f"Unexpected logits shape: {logits.shape}"
    )

    assert probs.shape == (1, 10), (
        f"Unexpected probs shape: {probs.shape}"
    )

    assert value.shape == (1,), (
        f"Unexpected value shape: {value.shape}"
    )

    assert torch.isfinite(logits).all(), (
        "Raw logits contain NaN or Inf"
    )

    assert torch.isfinite(probs).all(), (
        "Probabilities contain NaN or Inf"
    )

    assert torch.isfinite(value).all(), (
        "Value contains NaN or Inf"
    )

    assert torch.allclose(
        probs.sum(dim=1),
        torch.ones(1, device=probs.device),
        atol=1e-6,
    ), "Probabilities do not sum to 1"

    assert probs[0, 4].item() == 0.0, (
        "EvadeBack should be masked"
    )

    assert probs[0, 5].item() == 0.0, (
        "Retreat should be masked"
    )

    assert action_mask[action_id], (
        f"Sampled illegal action: {action_name}"
    )

    print()
    print("Actor-Critic inference smoke test: PASS")


if __name__ == "__main__":
    main()