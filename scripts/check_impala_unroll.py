import numpy as np
from pathlib import Path

from impala_unroll import build_unrolls

ROLLOUT_DIR = Path("data/rollouts/rollouts_bc_v2")

TIME_FIELDS = [
    "frames",
    "extra",
    "proposed_action_id",
    "final_action_id",
    "selected_reward",
    "reward_priority",
    "reward_high",
    "reward_medium",
    "reward_low",
    "done",
    "probs",
    "behavior_probs",
    "behavior_log_prob",
    "valid_mask",
]

def status(passed, total):
    if passed == total:
        return "OK"

    return f"FAIL ({passed}/{total})"

def check_unrolls(unrolls, unroll_length=20):
    check_pass_counts = {
        "time_length": 0,
        "valid_mask": 0,
        "terminal_boundary": 0,
        "bootstrap_valid": 0,
        "behavior_log_prob": 0,
        "raw_prob_sum": 0,
        "behavior_prob_sum": 0,
    }

    passed_unrolls = 0
    failed_unrolls = 0

    for i, unroll in enumerate(unrolls):
        errors = []
        time_length_ok = True
        for field in TIME_FIELDS:
            if len(unroll[field]) != unroll_length:
                time_length_ok = False
                errors.append(
                    f"field={field} has length "
                    f"{len(unroll[field])}, "
                    f"expected {unroll_length}"
                )
        if time_length_ok:
            check_pass_counts["time_length"] += 1

        valid_steps = int(unroll["valid_mask"].sum())

        expected_mask = np.zeros(
            unroll_length,
            dtype=np.float32,
        )

        expected_mask[:valid_steps] = 1.0

        if np.array_equal(unroll["valid_mask"], expected_mask):
            check_pass_counts["valid_mask"] += 1
        else:
            errors.append(
                f"valid_mask does not match expected mask for unroll {i}"
            )

        valid_steps = int(unroll["valid_mask"].sum())
        valid_done = unroll["done"][:valid_steps]
        if np.all(valid_done[:-1] == 0):
            check_pass_counts["terminal_boundary"] += 1
        else:
            errors.append(
                "done has non-zero value before "
                "the last valid step"
            )

        terminal = (valid_steps >0 and unroll["done"][valid_steps - 1] == 1)
        bootstrap_ok = (
            (terminal and unroll["bootstrap_valid"] == 0) or
            (not terminal and unroll["bootstrap_valid"] == 1)
        )
        if bootstrap_ok:
            check_pass_counts["bootstrap_valid"] += 1
        else:
            errors.append(
                f"terminal={terminal}, "
                f"but bootstrap_valid="
                f"{unroll['bootstrap_valid']}"
            )

        proposed = unroll["proposed_action_id"][:valid_steps]
        probs = unroll["probs"][:valid_steps]
        rows = np.arange(valid_steps)
        expected_action_probs = probs[rows, proposed]
        expected_log_probs = np.log(np.clip(expected_action_probs, 1e-8, 1.0))
        if np.allclose(
            unroll["behavior_log_prob"][:valid_steps],
            expected_log_probs,
            atol=1e-5,
        ): 
            check_pass_counts["behavior_log_prob"] += 1
        else:
            errors.append(
                "behavior_log_prob does not match "
                "raw probs of proposed actions"
            )

        raw_sums = unroll["probs"][:valid_steps].sum(axis=1)
        behavior_sums = unroll["behavior_probs"][:valid_steps].sum(axis=1)

        if np.allclose(raw_sums, 1.0, atol=1e-5):
            check_pass_counts["raw_prob_sum"] += 1
        else:
            errors.append(
                "raw action probabilities do not sum to 1"
            )
        if np.allclose(behavior_sums, 1.0, atol=1e-5):
            check_pass_counts["behavior_prob_sum"] += 1
        else:
            errors.append(
                "behavior action probabilities do not sum to 1"
            )

        if not errors:
            passed_unrolls += 1
            print(f"Unroll {i}: OK")
        else:
            failed_unrolls += 1

            print(f"Unroll {i}: FAIL")

            for error in errors:
                print(f"  - {error}")

        total_unrolls = len(unrolls)

        print("\n===== Summary =====")

        print(f"Total unrolls: {total_unrolls}")
        print(f"Passed: {passed_unrolls}")
        print(f"Failed: {failed_unrolls}")

        print(
            f"Time length: "
            f"{check_pass_counts['time_length']}"
            f"/{total_unrolls}"
        )

        print(
            f"Valid masks: "
            f"{check_pass_counts['valid_mask']}"
            f"/{total_unrolls}"
        )

        print(
            f"Terminal boundaries: "
            f"{check_pass_counts['terminal_boundary']}"
            f"/{total_unrolls}"
        )

        print(
            f"Bootstrap validity: "
            f"{check_pass_counts['bootstrap_valid']}"
            f"/{total_unrolls}"
        )

        print(
            f"Behavior log probs: "
            f"{check_pass_counts['behavior_log_prob']}"
            f"/{total_unrolls}"
        )

        print(
            f"Raw probability sums: "
            f"{check_pass_counts['raw_prob_sum']}"
            f"/{total_unrolls}"
        )

        print(
            f"Behavior probability sums: "
            f"{check_pass_counts['behavior_prob_sum']}"
            f"/{total_unrolls}"
        )

        print(
            "Time length:",
            status(
                check_pass_counts["time_length"],
                total_unrolls,
            ),
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
    check_unrolls(unrolls, unroll_length=20)

    print(
        "\n===== IMPALA Unroll "
        "Integrity Check ====="
    )

    check_unrolls(
        unrolls,
        unroll_length=20,
    )

if __name__ == "__main__":
    main()

    