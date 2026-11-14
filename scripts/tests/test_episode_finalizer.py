
"""Regression tests for Step 4-2C-3B Episode Finalization Extraction."""

import threading

import numpy as np
import pytest
import torch

import transition_tracker as tracker_module
from constant import BOSS_MAX_HP, PLAYER_MAX_HP
from episode_finalizer import (
    build_terminal_event_audit_snapshot,
    capture_terminal_combat_state,
    finalize_episode,
)
from episode_metrics import EpisodeEventAudit, PendingCombatEvents
from ground_truth_audit import GroundTruthAudit
from observation_builder import ACTION_NAME_TO_ID
from runtime.types import ActionSource, RuntimeAction
from transition_tracker import TransitionTracker


def make_policy_cache(action="Advance", frame_id=8):
    """Minimal valid Policy cache for the real rollout_logger."""
    num_actions = len(ACTION_NAME_TO_ID)

    return {
        "frames": torch.zeros((1, 3, 2, 2, 2)),
        "extra": torch.zeros((1, 24)),
        "logits": torch.zeros((1, num_actions)),
        "probs": torch.full((1, num_actions), 1.0 / num_actions),
        "behavior_probs": np.full(
            num_actions, 1.0 / num_actions, dtype=np.float32
        ),
        "proposed_action": action,
        "final_action": action,
        "info": {
            "visible": 1,
            "phase": "track",
            "search_hint": "center",
            "motion": 0.0,
        },
        "pol_state": {"hold_until_frame": frame_id + 1},
        "frame_id_end": frame_id,
        "fire_frame": frame_id + 1,
        "ue_att1_start": False,
        "ue_att1_end": False,
        "ue_att2_start": False,
        "ue_att2_end": False,
        "ue_boss_hit_count": 0,
        "ue_player_hit_count": 0,
        "ue_boss_damage": 0.0,
        "ue_player_damage": 0.0,
        "ue_episode_done": False,
        "ue_episode_result": 0,
        "action_mask": np.ones(num_actions, dtype=np.bool_),
        "actor_policy_step": 42,
    }


@pytest.fixture
def env(monkeypatch):
    """Real Tracker and rewards, without writing NPZ files."""
    event_audit = EpisodeEventAudit(
        player_max_hp=PLAYER_MAX_HP,
        boss_max_hp=BOSS_MAX_HP,
    )
    pending = PendingCombatEvents()
    ground_truth = GroundTruthAudit()
    tracker = TransitionTracker(
        actor_id="actor_test",
        event_audit=event_audit,
    )
    event_lock = threading.Lock()
    saved = []

    def fake_flush(buffer, actor_id, bootstrap_cathe=None):
        saved.append({
            "actor_id": actor_id,
            "steps": list(buffer),
            "bootstrap_cache": bootstrap_cathe,
        })
        buffer.clear()

    monkeypatch.setattr(
        tracker_module,
        "flush_rollout_buffer",
        fake_flush,
    )
    monkeypatch.setattr(
        tracker_module,
        "ROLLOUT_SAVE_EVERY",
        999,
    )

    return {
        "tracker": tracker,
        "event_audit": event_audit,
        "pending": pending,
        "gt": ground_truth,
        "lock": event_lock,
        "saved": saved,
    }


def finish(env, result=1):
    """Call the shared Episode Finalizer."""
    return finalize_episode(
        tracker=env["tracker"],
        event_audit=env["event_audit"],
        pending_combat=env["pending"],
        ground_truth_audit=env["gt"],
        event_lock=env["lock"],
        episode_result=result,
        policy_step=42,
    )


# =========================================================
# 1. Three terminal call sites share identical semantics
# =========================================================

@pytest.mark.parametrize(
    "terminal_path",
    ["normal", "early", "disconnect"],
)
def test_terminal_paths_use_same_finalization(terminal_path, env):
    # terminal_path represents the caller, not a finalizer argument.
    tracker = env["tracker"]
    tracker.record_policy_sent(make_policy_cache("Advance"))

    summary = finish(env, result=1)

    assert len(env["saved"]) == 1
    assert len(env["saved"][0]["steps"]) == 1

    step = env["saved"][0]["steps"][0]

    assert int(step["done"]) == 1
    assert int(step["final_action_id"]) == ACTION_NAME_TO_ID["Advance"]
    assert float(step["reward_high"]) == pytest.approx(5.0)

    assert summary["steps"] == 1
    assert summary["selected_return"] == pytest.approx(5.0)
    assert summary["result"] == "Boss Win"
    assert summary["policy_step"] == 42


# =========================================================
# 2. Terminal reward must reflect actual episode result
# =========================================================

@pytest.mark.parametrize(
    "result,expected_reward,expected_label",
    [
        (1, 5.0, "Boss Win"),
        (-1, -5.0, "Boss Loss"),
    ],
)
def test_terminal_reward_win_and_loss(
    env, result, expected_reward, expected_label
):
    env["tracker"].record_policy_sent(make_policy_cache())

    summary = finish(env, result=result)
    step = env["saved"][0]["steps"][0]

    assert int(step["done"]) == 1
    assert int(step["ue_episode_result"]) == result
    assert float(step["reward_high"]) == pytest.approx(expected_reward)

    assert summary["result"] == expected_label
    assert summary["selected_return"] == pytest.approx(expected_reward)


# =========================================================
# 3. Pending combat is consumed and attributed once
# =========================================================

def test_pending_combat_attributed_to_last_policy(env):
    tracker = env["tracker"]
    audit = env["event_audit"]
    pending = env["pending"]

    tracker.record_policy_sent(make_policy_cache("Advance"))

    # Raw UE event audit
    audit.player_hits_on_boss_raw = 2
    audit.boss_hits_on_player_raw = 1
    audit.record_player_health(90.0)
    audit.record_boss_health(475.0)

    # Not yet consumed by the inference loop
    pending.boss_hit_count = 2
    pending.player_hit_count = 1
    pending.boss_damage = 25.0
    pending.player_damage = 10.0

    summary = finish(env, result=1)
    step = env["saved"][0]["steps"][0]

    assert int(step["ue_boss_hit_count"]) == 2
    assert int(step["ue_player_hit_count"]) == 1
    assert float(step["ue_boss_damage"]) == pytest.approx(25.0)
    assert float(step["ue_player_damage"]) == pytest.approx(10.0)

    expected_medium = (
        10.0 / PLAYER_MAX_HP
        - 25.0 / BOSS_MAX_HP
    )

    assert float(step["reward_medium"]) == pytest.approx(expected_medium)
    assert float(step["reward_high"]) == pytest.approx(5.0)

    # High priority wins when terminal is present.
    assert summary["selected_return"] == pytest.approx(5.0)

    assert summary["raw_player_hits_on_boss"] == 2
    assert summary["raw_boss_hits_on_player"] == 1
    assert summary["player_damage_to_boss_raw"] == pytest.approx(25.0)
    assert summary["boss_damage_to_player_raw"] == pytest.approx(10.0)

    assert pending.boss_hit_count == 0
    assert pending.player_hit_count == 0
    assert pending.boss_damage == 0.0
    assert pending.player_damage == 0.0


# =========================================================
# 4. Anchor is retained when no active Policy cache exists
# =========================================================

def test_terminal_uses_previous_valid_anchor(env):
    tracker = env["tracker"]

    tracker.record_policy_sent(make_policy_cache("Advance"))
    tracker.rotate_at_decision_boundary()

    # No active Policy, e.g. Runtime Recovery is executing.
    env["pending"].player_hit_count = 1
    env["pending"].player_damage = 15.0

    finish(env, result=-1)

    assert len(env["saved"]) == 1
    assert len(env["saved"][0]["steps"]) == 1

    step = env["saved"][0]["steps"][0]

    assert int(step["final_action_id"]) == ACTION_NAME_TO_ID["Advance"]
    assert int(step["done"]) == 1
    assert float(step["reward_high"]) == pytest.approx(-5.0)

    # Runtime interval damage is not assigned to previous Policy.
    assert float(step["reward_medium"]) == pytest.approx(0.0)
    assert int(step["ue_player_hit_count"]) == 0

    assert env["event_audit"].unattributed_boss_hits_on_player == 1
    assert (
        env["event_audit"].unattributed_boss_damage_to_player
        == pytest.approx(15.0)
    )


# =========================================================
# 5. Multiple Policy transitions keep the correct done flags
# =========================================================

def test_multiple_policy_steps_only_last_is_terminal(env):
    tracker = env["tracker"]

    tracker.record_policy_sent(make_policy_cache("Advance", 8))
    tracker.rotate_at_decision_boundary()

    tracker.record_policy_sent(make_policy_cache("StrafeRight", 12))

    finish(env, result=1)

    assert len(env["saved"]) == 1

    steps = env["saved"][0]["steps"]
    assert len(steps) == 2

    assert [int(x["done"]) for x in steps] == [0, 1]
    assert [int(x["final_action_id"]) for x in steps] == [
        ACTION_NAME_TO_ID["Advance"],
        ACTION_NAME_TO_ID["StrafeRight"],
    ]
    assert [float(x["reward_high"]) for x in steps] == pytest.approx(
        [0.0, 5.0]
    )


# =========================================================
# 6. Zero valid Policy steps must not produce fake training
# =========================================================

def test_zero_policy_terminal_does_not_invent_step(env, capsys):
    summary = finish(env, result=1)

    assert env["saved"] == []
    assert summary["steps"] == 0
    assert summary["selected_return"] == pytest.approx(0.0)
    assert summary["result"] == "Boss Win"

    output = capsys.readouterr().out
    assert "without any trainable policy transition" in output


# =========================================================
# 7. Pending combat without Policy becomes unattributed
# =========================================================

def test_terminal_combat_without_policy_unattributed(env):
    pending = env["pending"]

    pending.boss_hit_count = 2
    pending.player_hit_count = 1
    pending.boss_damage = 30.0
    pending.player_damage = 10.0

    summary = finish(env, result=-1)

    assert env["saved"] == []
    assert summary["steps"] == 0

    assert summary["unattributed_player_hits_on_boss"] == 2
    assert summary["unattributed_boss_hits_on_player"] == 1
    assert summary["unattributed_player_damage_to_boss"] == pytest.approx(
        30.0
    )
    assert summary["unattributed_boss_damage_to_player"] == pytest.approx(
        10.0
    )


# =========================================================
# 8. Ground Truth action must be aborted on terminal
# =========================================================

def test_terminal_aborts_inflight_ground_truth_action(env):
    gt = env["gt"]

    action = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    gt.begin_action(
        action="Advance",
        source="policy",
        expected_runtime_action=action,
    )

    env["tracker"].record_policy_sent(make_policy_cache("Advance"))

    finish(env, result=1)

    assert gt.current is None
    assert len(gt.records) == 1
    assert gt.records[0].completed is False
    assert gt.records[0].aborted_reason == "episode_terminal"
    assert gt.action_mismatch == 0


# =========================================================
# 9. Already finished Ground Truth must remain untouched
# =========================================================

def test_terminal_does_not_abort_completed_ground_truth(env):
    gt = env["gt"]

    action = RuntimeAction(
        "Advance",
        0.0,
        ActionSource.POLICY,
    )

    gt.begin_action(
        action="Advance",
        source="policy",
        expected_runtime_action=action,
    )
    gt.finish_action(action)

    env["tracker"].record_policy_sent(make_policy_cache())

    finish(env, result=1)

    assert len(gt.records) == 1
    assert gt.records[0].completed is True
    assert gt.records[0].aborted_reason is None
    assert gt.action_mismatch == 0


# =========================================================
# 10. Snapshot preserves raw and unattributed data
# =========================================================

def test_terminal_snapshot_preserves_audit_fields(env):
    audit = env["event_audit"]

    audit.player_hits_on_boss_raw = 4
    audit.boss_hits_on_player_raw = 3
    audit.unattributed_player_hits_on_boss = 1
    audit.unattributed_boss_hits_on_player = 2
    audit.unattributed_player_damage_to_boss = 12.0
    audit.unattributed_boss_damage_to_player = 8.0

    audit.record_player_health(70.0)
    audit.record_boss_health(450.0)

    state = capture_terminal_combat_state(
        event_lock=env["lock"],
        event_audit=audit,
        pending_combat=env["pending"],
    )

    snapshot = build_terminal_event_audit_snapshot(
        event_audit=audit,
        terminal_combat_state=state,
    )

    assert snapshot["player_hits_on_boss_raw"] == 4
    assert snapshot["boss_hits_on_player_raw"] == 3
    assert snapshot["unattributed_player_hits_on_boss"] == 1
    assert snapshot["unattributed_boss_hits_on_player"] == 2

    assert snapshot["unattributed_player_damage_to_boss"] == pytest.approx(
        12.0
    )
    assert snapshot["unattributed_boss_damage_to_player"] == pytest.approx(
        8.0
    )

    assert snapshot["player_damage_to_boss_raw"] == pytest.approx(50.0)
    assert snapshot["boss_damage_to_player_raw"] == pytest.approx(30.0)

    assert snapshot["final_player_health"] == pytest.approx(70.0)
    assert snapshot["final_boss_health"] == pytest.approx(450.0)

    assert snapshot["player_max_hp"] == PLAYER_MAX_HP
    assert snapshot["boss_max_hp"] == BOSS_MAX_HP


# =========================================================
# 11. Terminal capture must consume PendingCombatEvents
# =========================================================

def test_terminal_capture_consumes_pending_only_once(env):
    pending = env["pending"]

    pending.boss_hit_count = 2
    pending.player_hit_count = 3
    pending.boss_damage = 40.0
    pending.player_damage = 20.0

    first = capture_terminal_combat_state(
        event_lock=env["lock"],
        event_audit=env["event_audit"],
        pending_combat=pending,
    )

    second = capture_terminal_combat_state(
        event_lock=env["lock"],
        event_audit=env["event_audit"],
        pending_combat=pending,
    )

    assert first["pending_boss_hits"] == 2
    assert first["pending_player_hits"] == 3
    assert first["pending_boss_damage"] == pytest.approx(40.0)
    assert first["pending_player_damage"] == pytest.approx(20.0)

    assert second["pending_boss_hits"] == 0
    assert second["pending_player_hits"] == 0
    assert second["pending_boss_damage"] == pytest.approx(0.0)
    assert second["pending_player_damage"] == pytest.approx(0.0)
