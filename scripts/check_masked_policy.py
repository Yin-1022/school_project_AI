import torch
import numpy as np

from policy_inference import apply_action_mask

def masked_test():
    logits = torch.tensor([
        [1.0, 2.0, 10.0, 0.5]
    ])

    action_mask = np.array([
        True,
        True,
        False,
        True,
    ])

    masked_logits = apply_action_mask(logits, action_mask)
    probs = torch.softmax(masked_logits, dim=-1)

    # 1. probability 總和仍然是 1
    assert torch.allclose(
        probs.sum(dim=-1),
        torch.ones(1),
    )

    # 2. 被 mask 的 action probability 必須是 0
    assert probs[0, 2].item() == 0.0

    # 3. raw logit 最大的是 action 2，
    #    但 mask 後絕對不能選它
    action_id = probs.argmax(dim=-1).item()

    assert action_id != 2
    assert action_id == 1

    # 4. sample 很多次也不能出現 masked action
    samples = torch.multinomial(
        probs,
        num_samples=1000,
        replacement=True,
    )

    assert not torch.any(
        samples == 2
    )

    print("Probability sum: OK")
    print("Masked probability = 0: OK")
    print("Masked argmax blocked: OK")
    print("Masked sampling blocked: OK")

def no_mask_test():
    logits = torch.tensor([
        [1.0, 2.0, 10.0, 0.5]
    ])

    all_valid_mask = np.array([
        True,
        True,
        True,
        True,
    ])

    all_valid_logits = apply_action_mask(
        logits,
        all_valid_mask,
    )

    masked_probs = torch.softmax(
        all_valid_logits,
        dim=-1,
    )

    raw_probs = torch.softmax(
        logits,
        dim=-1,
    )

    assert torch.allclose(
        masked_probs,
        raw_probs,
    )

    print("No mask test: OK")

def main():
    masked_test()
    no_mask_test()

if __name__ == "__main__":
    main()