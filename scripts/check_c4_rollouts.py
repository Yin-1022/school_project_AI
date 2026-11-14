
"""Inspect UE smoke-test rollout_v4 NPZ artifacts."""

import argparse
from pathlib import Path

import numpy as np


REQUIRED = {
    "done",
    "ue_episode_done",
    "ue_episode_result",
    "reward_high",
    "reward_medium",
    "reward_low",
    "ue_player_damage",
    "ue_boss_damage",
    "proposed_action_id",
    "final_action_id",
    "behavior_probs",
    "action_mask",
    "actor_policy_step",
}


def check_file(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED - set(data.files)
        assert not missing, f"{path.name}: missing {sorted(missing)}"

        done = data["done"]
        result = data["ue_episode_result"]
        high = data["reward_high"]
        medium = data["reward_medium"]
        low = data["reward_low"]
        player_damage = data["ue_player_damage"]
        boss_damage = data["ue_boss_damage"]

        n = len(done)
        assert n > 0, f"{path.name}: empty rollout"

        for key in REQUIRED:
            assert len(data[key]) == n, (
                f"{path.name}: {key} length mismatch"
            )

        assert np.isin(done, [0, 1]).all()
        assert np.isin(result, [-1, 0, 1]).all()

        for key in (
            "reward_high",
            "reward_medium",
            "reward_low",
            "behavior_probs",
            "ue_player_damage",
            "ue_boss_damage",
        ):
            assert np.isfinite(data[key]).all(), (
                f"{path.name}: non-finite {key}"
            )

        final_ids = data["final_action_id"]
        assert ((final_ids >= 0) & (final_ids < 10)).all()

        # Current IMPALA medium reward formula.
        expected_medium = (
            player_damage / 100.0 - boss_damage / 500.0
        )
        assert np.allclose(
            medium, expected_medium, atol=1e-5
        ), f"{path.name}: medium reward mismatch"

        terminal_indices = np.flatnonzero(done == 1)
        assert len(terminal_indices) <= 1, (
            f"{path.name}: multiple terminal steps"
        )

        if len(terminal_indices) == 1:
            idx = int(terminal_indices[0])

            assert idx == n - 1, (
                f"{path.name}: terminal is not last"
            )
            assert int(result[idx]) in (-1, 1)
            assert int(data["ue_episode_done"][idx]) == 1
            assert np.isclose(
                high[idx], 5.0 * result[idx]
            ), f"{path.name}: incorrect terminal reward"

        nonterminal = done == 0
        assert np.all(result[nonterminal] == 0), (
            f"{path.name}: result set on nonterminal step"
        )
        assert np.allclose(high[nonterminal], 0.0), (
            f"{path.name}: unexpected nonterminal high reward"
        )

        if "bootstrap_frames" in data.files:
            assert not done.any(), (
                f"{path.name}: terminal rollout has bootstrap"
            )
            assert "bootstrap_extra" in data.files

        terminal_count = len(terminal_indices)

        print(
            f"[PASS] {path.name} "
            f"steps={n} terminals={terminal_count} "
            f"medium_sum={float(medium.sum()):+.4f}"
        )

        return n, terminal_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/rollouts/rollouts_damage_v1"),
    )
    parser.add_argument(
        "--require-terminal",
        action="store_true",
    )
    args = parser.parse_args()

    files = sorted(args.dir.glob("rollout_v4_*.npz"))
    if not files:
        parser.error(f"No rollout NPZ found in {args.dir}")

    total_steps = 0
    total_terminals = 0

    for path in files:
        steps, terminals = check_file(path)
        total_steps += steps
        total_terminals += terminals

    if args.require_terminal:
        assert total_terminals > 0, (
            "No terminal transition found"
        )

    print(
        f"\n[SUMMARY] files={len(files)} "
        f"steps={total_steps} "
        f"terminals={total_terminals}"
    )


if __name__ == "__main__":
    main()


# import numpy as np

# path = (
#     "data/rollouts/rollouts_damage_v1/"
#     "rollout_v4_train_masked_actor_0_1794645807760.npz"
# )

# with np.load(path, allow_pickle=False) as d:
#     idx = np.flatnonzero(d["done"] == 1)[0]

#     for key in (
#         "final_action_id",
#         "ue_episode_result",
#         "ue_boss_damage",
#         "ue_player_damage",
#         "reward_high",
#         "reward_medium",
#         "reward_low",
#     ):
#         print(key, d[key][idx])