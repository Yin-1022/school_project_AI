from pathlib import Path

import numpy as np
import torch

from constant import ROLLOUT_DIR
from impala_loss import compute_impala_loss
from impala_unroll import build_unrolls
from vtrace import compute_importance_ratios


UNROLL_LENGTH = 20
ATOL = 1e-5
ZERO_ATOL = 1e-7


def find_latest_masked_rollout() -> Path:
    files = sorted(
        ROLLOUT_DIR.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(
            f"No rollout files found in {ROLLOUT_DIR}"
        )

    # 舊 rollout 可能沒有 action_mask，
    # 所以從最新往前找第一個真的包含 mask 的 rollout。
    for path in reversed(files):
        with np.load(path, allow_pickle=False) as data:
            if "action_mask" in data.files:
                return path

    raise RuntimeError(
        "No rollout containing action_mask was found. "
        "Collect a new rollout with the Step 13 masked actor first."
    )


def check_rollout_mask(data):
    required = [
        "logits",
        "probs",
        "proposed_action_id",
        "action_mask",
    ]

    missing = [
        field
        for field in required
        if field not in data.files
    ]

    if missing:
        raise AssertionError(
            f"Rollout missing masked-policy fields: {missing}"
        )

    logits = np.asarray(
        data["logits"],
        dtype=np.float32,
    )

    probs = np.asarray(
        data["probs"],
        dtype=np.float32,
    )

    actions = np.asarray(
        data["proposed_action_id"],
        dtype=np.int64,
    )

    action_mask = np.asarray(
        data["action_mask"],
        dtype=np.bool_,
    )

    num_steps = len(actions)
    num_actions = probs.shape[-1]

    assert logits.shape == (
        num_steps,
        num_actions,
    )

    assert probs.shape == (
        num_steps,
        num_actions,
    )

    assert action_mask.shape == (
        num_steps,
        num_actions,
    )

    assert action_mask.dtype == np.bool_

    assert np.all(
        (actions >= 0)
        & (actions < num_actions)
    )

    # 每個真實 transition 至少要有一個合法 action。
    assert np.all(
        action_mask.any(axis=1)
    )

    rows = np.arange(num_steps)

    # Actor 實際 sample 出來的 proposed action
    # 必須在當時的 mask 中是合法的。
    assert np.all(
        action_mask[
            rows,
            actions,
        ]
    )

    # Masked policy distribution 仍必須 sum=1。
    assert np.allclose(
        probs.sum(axis=1),
        1.0,
        atol=ATOL,
    )

    # 所有 illegal action 機率都必須是 0。
    illegal_probs = probs[
        ~action_mask
    ]

    if illegal_probs.size > 0:
        assert np.all(
            np.abs(illegal_probs)
            <= ZERO_ATOL
        )

    # --------------------------------------------------
    # 驗 Actor rollout：
    #
    # raw logits
    #     + stored action_mask
    #     ↓
    # masked logits
    #     ↓ softmax
    # 應該 == rollout 裡存的 probs
    # --------------------------------------------------

    logits_tensor = torch.from_numpy(
        logits
    )

    action_mask_tensor = torch.from_numpy(
        action_mask
    )

    masked_logits = logits_tensor.masked_fill(
        ~action_mask_tensor,
        float("-inf"),
    )

    recomputed_probs = torch.softmax(
        masked_logits,
        dim=-1,
    )

    assert torch.isfinite(
        recomputed_probs
    ).all()

    assert torch.allclose(
        recomputed_probs,
        torch.from_numpy(probs),
        atol=ATOL,
        rtol=ATOL,
    )

    masked_step_count = int(
        np.any(
            ~action_mask,
            axis=1,
        ).sum()
    )

    masked_slot_count = int(
        (~action_mask).sum()
    )

    print(
        "rollout action_mask:",
        action_mask.shape,
        f"dtype={action_mask.dtype}",
    )

    print(
        "masked transitions:",
        f"{masked_step_count}/{num_steps}",
    )

    print(
        "masked action slots:",
        masked_slot_count,
    )

    print("NPZ mask schema: OK")
    print("Proposed actions legal: OK")
    print("Actor probability sums: OK")
    print(
        "Illegal actor probabilities are zero: OK"
    )
    print(
        "Raw logits + mask reproduce actor probs: OK"
    )

    if masked_step_count == 0:
        print(
            "WARNING: this rollout contains no transition "
            "where an action was actually masked. "
            "The data path is valid, but collect a rollout "
            "that hits a cooldown or visible-track restriction "
            "to exercise a real masking event."
        )

    return {
        "logits": logits_tensor,
        "probs": torch.from_numpy(probs),
        "actions": torch.from_numpy(actions),
        "num_steps": num_steps,
        "num_actions": num_actions,
        "masked_logits": masked_logits,
    }


def check_unroll_masks(
    data,
    num_actions,
):
    has_behavior_probs = (
        "behavior_probs" in data.files
    )

    unrolls = build_unrolls(
        data,
        unroll_length=UNROLL_LENGTH,
        has_behavior_probs=has_behavior_probs,
    )

    assert unrolls, (
        "No IMPALA unrolls were produced"
    )

    padded_unroll_count = 0

    for unroll in unrolls:
        action_mask = np.asarray(
            unroll["action_mask"],
            dtype=np.bool_,
        )

        valid_mask = np.asarray(
            unroll["valid_mask"]
        ).astype(bool)

        actions = np.asarray(
            unroll["proposed_action_id"],
            dtype=np.int64,
        )

        probs = np.asarray(
            unroll["probs"],
            dtype=np.float32,
        )

        behavior_log_prob = np.asarray(
            unroll["behavior_log_prob"],
            dtype=np.float32,
        )

        assert action_mask.shape == (
            UNROLL_LENGTH,
            num_actions,
        )

        assert valid_mask.shape == (
            UNROLL_LENGTH,
        )

        valid_steps = int(
            valid_mask.sum()
        )

        assert valid_steps > 0

        valid_action_mask = (
            action_mask[valid_mask]
        )

        valid_actions = (
            actions[valid_mask]
        )

        valid_probs = (
            probs[valid_mask]
        )

        valid_rows = np.arange(
            valid_steps
        )

        # 每個有效 transition 至少一個合法 action。
        assert np.all(
            valid_action_mask.any(axis=1)
        )

        # proposed action 必須合法。
        assert np.all(
            valid_action_mask[
                valid_rows,
                valid_actions,
            ]
        )

        # ------------------------------------------------
        # behavior_log_prob
        # =
        # log μ_masked(proposed_action | state)
        # ------------------------------------------------

        expected_behavior_log_prob = np.log(
            np.clip(
                valid_probs[
                    valid_rows,
                    valid_actions,
                ],
                1e-8,
                1.0,
            )
        )

        assert np.allclose(
            behavior_log_prob[
                valid_mask
            ],
            expected_behavior_log_prob,
            atol=ATOL,
            rtol=ATOL,
        )

        # ------------------------------------------------
        # Terminal padding：
        #
        # action_mask padding 必須 all-True。
        #
        # 不然：
        # all False
        # → logits 全 -inf
        # → softmax = NaN
        # ------------------------------------------------

        padding_mask = ~valid_mask

        if np.any(padding_mask):
            padded_unroll_count += 1

            assert np.all(
                action_mask[
                    padding_mask
                ]
            )

        assert np.allclose(
            valid_probs.sum(axis=1),
            1.0,
            atol=ATOL,
        )

        illegal_valid_probs = (
            valid_probs[
                ~valid_action_mask
            ]
        )

        if illegal_valid_probs.size > 0:
            assert np.all(
                np.abs(
                    illegal_valid_probs
                )
                <= ZERO_ATOL
            )

    print(
        "unrolls checked:",
        len(unrolls),
    )

    print(
        "Unroll action_mask shape: OK"
    )

    print(
        "Valid unroll actions legal: OK"
    )

    print(
        "behavior_log_prob uses masked μ: OK"
    )

    if padded_unroll_count > 0:
        print(
            "Terminal padding action masks "
            "are all-True: OK"
        )
    else:
        print(
            "Terminal padding action-mask check: "
            "SKIPPED "
            "(no padded unroll in this rollout)"
        )


def check_actor_learner_symmetry(
    rollout_check,
):
    masked_logits = (
        rollout_check["masked_logits"]
    )

    probs = rollout_check["probs"]
    actions = rollout_check["actions"]

    rows = torch.arange(
        rollout_check["num_steps"]
    )

    # Actor 當時真正的 masked μ(a_t | s_t)
    behavior_action_probs = probs[
        rows,
        actions,
    ]

    behavior_log_prob = torch.log(
        torch.clamp(
            behavior_action_probs,
            min=1e-8,
            max=1.0,
        )
    )

    # --------------------------------------------------
    # 假設 Learner logits 跟 Actor raw logits 完全相同，
    # 又套同一份 stored mask：
    #
    # π_masked == μ_masked
    #
    # 所以：
    #
    # rho = π / μ = 1
    # --------------------------------------------------

    (
        target_action_log_prob,
        log_rhos,
        rhos,
    ) = compute_importance_ratios(
        target_logits=masked_logits,
        actions=actions,
        behavior_log_prob=behavior_log_prob,
    )

    assert torch.isfinite(
        target_action_log_prob
    ).all()

    assert torch.isfinite(
        log_rhos
    ).all()

    assert torch.isfinite(
        rhos
    ).all()

    assert torch.allclose(
        rhos,
        torch.ones_like(rhos),
        atol=ATOL,
        rtol=ATOL,
    )

    print(
        "same logits + same stored mask "
        "-> rho ~= 1: OK"
    )

    print(
        "rho range:",
        f"{rhos.min().item():.6f}",
        "~",
        f"{rhos.max().item():.6f}",
    )

    # --------------------------------------------------
    # Entropy safety：
    #
    # masked logits 裡有 -inf，
    # impala_loss 必須仍然是 finite。
    # --------------------------------------------------

    zeros = torch.zeros(
        rollout_check["num_steps"],
        dtype=torch.float32,
    )

    valid_mask = torch.ones_like(
        zeros
    )

    losses = compute_impala_loss(
        target_logits=masked_logits,
        target_action_log_prob=(
            target_action_log_prob
        ),
        values=zeros,
        vs=zeros,
        pg_advantages=zeros,
        valid_mask=valid_mask,
    )

    for name, value in losses.items():
        assert torch.isfinite(value), (
            f"Non-finite {name}: {value}"
        )

    print(
        "Masked entropy/loss finite: OK"
    )


def main():
    path = find_latest_masked_rollout()

    print(
        f"loading: {path}"
    )

    print(
        "\n===== Masked IMPALA Check ====="
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        rollout_check = (
            check_rollout_mask(data)
        )

        print(
            "\n===== Unroll Mask Check ====="
        )

        check_unroll_masks(
            data,
            num_actions=(
                rollout_check[
                    "num_actions"
                ]
            ),
        )

        print(
            "\n===== Actor / Learner "
            "Mask Symmetry Check ====="
        )

        check_actor_learner_symmetry(
            rollout_check
        )

    print(
        "\n===== Masked IMPALA: OK ====="
    )

if __name__ == "__main__":
    main()