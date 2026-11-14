
"""Regression tests for Step 4-2C-3C."""

import ast
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

import transition_tracker as tracker_module

from constant import PLAYER_MAX_HP, BOSS_MAX_HP
from episode_metrics import EpisodeEventAudit, PendingCombatEvents
from ground_truth_audit import GroundTruthAudit
from impala_event_coordinator import (
    handle_rejection_pulse,
    finalize_terminal_episode,
)
from observation_builder import ACTION_NAME_TO_ID
from runtime.adapters.impala_adapter import to_runtime_action
from runtime.controller import RuntimeController
from runtime.types import ActionSource
from transition_tracker import TransitionTracker


def make_cache(action="Advance"):
    n = len(ACTION_NAME_TO_ID)

    return {
        "frames": torch.zeros((1, 3, 2, 2, 2)),
        "extra": torch.zeros((1, 24)),
        "logits": torch.zeros((1, n)),
        "probs": torch.full((1, n), 1.0 / n),
        "behavior_probs": np.full(n, 1.0 / n, dtype=np.float32),
        "proposed_action": action,
        "final_action": action,
        "info": {
            "visible": 1,
            "phase": "track",
            "search_hint": "center",
            "motion": 0.0,
        },
        "pol_state": {"hold_until_frame": 10},
        "frame_id_end": 8,
        "fire_frame": 9,
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
        "action_mask": np.ones(n, dtype=np.bool_),
        "actor_policy_step": 42,
    }


@pytest.fixture
def env(monkeypatch):
    audit = EpisodeEventAudit(
        player_max_hp=PLAYER_MAX_HP,
        boss_max_hp=BOSS_MAX_HP,
    )
    tracker = TransitionTracker(
        actor_id="actor_test",
        event_audit=audit,
    )
    runtime = RuntimeController()
    gt = GroundTruthAudit()
    pending = PendingCombatEvents()
    lock = threading.Lock()

    state = {
        "episode_done_flag": True,
        "episode_result": -1,
        "notseeplayer_pulse": False,
    }

    saved = []

    def fake_flush(buffer, actor_id, bootstrap_cathe=None):
        saved.append(list(buffer))
        buffer.clear()

    monkeypatch.setattr(
        tracker_module, "flush_rollout_buffer", fake_flush
    )
    monkeypatch.setattr(
        tracker_module, "ROLLOUT_SAVE_EVERY", 999
    )

    return {
        "tracker": tracker,
        "runtime": runtime,
        "gt": gt,
        "audit": audit,
        "pending": pending,
        "lock": lock,
        "state": state,
        "saved": saved,
    }


def send_policy(env, action_name):
    action = to_runtime_action(action_name)

    env["runtime"].on_action_sent(action)
    env["tracker"].record_policy_sent(make_cache(action_name))

    env["gt"].begin_action(
        action=action_name,
        source="policy",
        expected_runtime_action=action,
    )

    return action


def reject(env, pulse=True):
    return handle_rejection_pulse(
        pulse=pulse,
        runtime=env["runtime"],
        tracker=env["tracker"],
        ground_truth_audit=env["gt"],
        event_lock=env["lock"],
    )


def finish(env, result=-1):
    return finalize_terminal_episode(
        runtime=env["runtime"],
        tracker=env["tracker"],
        event_audit=env["audit"],
        pending_combat=env["pending"],
        ground_truth_audit=env["gt"],
        event_lock=env["lock"],
        event_state=env["state"],
        episode_result=result,
        policy_step=42,
    )


def test_rejection_is_immediate_and_preserves_anchor(env):
    tracker = env["tracker"]

    tracker.record_policy_sent(make_cache("Advance"))
    tracker.rotate_at_decision_boundary()

    send_policy(env, "StrafeLeft")

    # The rejected action already accumulated some combat events.
    env["tracker"].record_frame_events(
        boss_hit_count=1,
        player_hit_count=0,
        boss_damage=20.0,
        player_damage=0.0,
        att1_start=False,
        att1_end=False,
        att2_start=False,
        att2_end=False,
        episode_done=False,
        episode_result=0,
    )

    assert reject(env) is True
    assert tracker._last_step_cache is None
    assert env["runtime"].fallback.pending is True

    assert env["gt"].records[-1].aborted_reason == (
        "notseeplayer_rejected"
    )
    assert (
        env["audit"].unattributed_player_damage_to_boss
        == pytest.approx(20.0)
    )

    finish(env, result=-1)

    assert len(env["saved"]) == 1
    assert len(env["saved"][0]) == 1

    step = env["saved"][0][0]
    assert int(step["final_action_id"]) == ACTION_NAME_TO_ID["Advance"]
    assert int(step["done"]) == 1
    assert float(step["reward_high"]) == pytest.approx(-5.0)
    assert float(step["reward_medium"]) == pytest.approx(0.0)


def test_non_strafe_notseeplayer_does_not_discard_policy(env):
    send_policy(env, "Advance")

    assert reject(env) is False

    assert env["tracker"]._last_step_cache is not None
    assert env["gt"].current is not None
    assert env["gt"].action_mismatch == 0
    assert env["runtime"].fallback.pending is False


def test_duplicate_rejection_pulse_does_not_abort_twice(env):
    send_policy(env, "StrafeLeft")

    assert reject(env) is True
    assert reject(env) is False

    assert len(env["gt"].records) == 1
    assert env["gt"].action_mismatch == 0
    assert env["audit"].unattributed_player_hits_on_boss == 0


def test_rejection_cache_mismatch_is_detected(env):
    runtime_action = to_runtime_action("StrafeLeft")

    env["runtime"].on_action_sent(runtime_action)
    env["tracker"].record_policy_sent(make_cache("StrafeRight"))

    with pytest.raises(RuntimeError, match="mismatch"):
        reject(env)

    # Do not silently discard a different Policy transition.
    assert env["tracker"]._last_step_cache is not None


def test_rejection_does_not_bypass_attack_guard(env):
    send_policy(env, "StrafeLeft")

    env["runtime"].on_attack_start("att1")

    assert reject(env) is True
    assert env["runtime"].can_send() is False
    assert env["runtime"].next_action() is None

    env["runtime"].on_attack_end("att1")

    fallback = env["runtime"].next_action()
    assert fallback is not None
    assert fallback.name == "Advance"
    assert fallback.source == ActionSource.RUNTIME_FALLBACK


@pytest.mark.parametrize(
    "terminal_path",
    ["early", "normal", "disconnect"],
)
def test_terminal_pending_rejection_drops_strafe(
    env, terminal_path
):
    tracker = env["tracker"]

    tracker.record_policy_sent(make_cache("Advance"))
    tracker.rotate_at_decision_boundary()

    send_policy(env, "StrafeRight")

    env["state"]["notseeplayer_pulse"] = True
    env["pending"].boss_hit_count = 1
    env["pending"].boss_damage = 30.0

    summary = finish(env, result=-1)

    assert len(env["saved"]) == 1
    assert len(env["saved"][0]) == 1

    step = env["saved"][0][0]
    assert int(step["final_action_id"]) == ACTION_NAME_TO_ID["Advance"]
    assert int(step["done"]) == 1
    assert float(step["reward_high"]) == pytest.approx(-5.0)

    assert summary["steps"] == 1
    assert summary["unattributed_player_hits_on_boss"] == 1
    assert (
        summary["unattributed_player_damage_to_boss"]
        == pytest.approx(30.0)
    )

    assert env["state"]["notseeplayer_pulse"] is False
    assert env["runtime"].execution.state.busy is False
    assert env["runtime"].fallback.pending is False

    # Preserve terminal latch until the next episode starts.
    assert env["state"]["episode_done_flag"] is True
    assert env["state"]["episode_result"] == -1


def test_terminal_without_rejection_keeps_valid_policy(env):
    send_policy(env, "Advance")

    summary = finish(env, result=1)

    assert summary["steps"] == 1
    step = env["saved"][0][0]

    assert int(step["done"]) == 1
    assert float(step["reward_high"]) == pytest.approx(5.0)
    assert env["runtime"].execution.state.busy is False


def test_terminal_with_no_policy_does_not_create_fake_step(env):
    summary = finish(env, result=-1)

    assert summary["steps"] == 0
    assert env["saved"] == []


@pytest.mark.parametrize("result", [0, 2, -2])
def test_invalid_terminal_result_rejected(env, result):
    with pytest.raises(ValueError, match="Invalid terminal"):
        finish(env, result=result)

    assert env["saved"] == []


def test_main_terminal_call_sites_use_correct_results():
    """Static integration guard against the C-3B scope regression."""

    main_path = (
        Path(__file__).resolve().parents[1] / "stream_infer.py"
    )

    tree = ast.parse(main_path.read_text(encoding="utf-8"))

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "finalize_terminal_episode"
        )
    ]

    calls.sort(key=lambda node: node.lineno)

    assert len(calls) == 3

    result_names = []
    for call in calls:
        keyword = next(
            kw for kw in call.keywords
            if kw.arg == "episode_result"
        )
        assert isinstance(keyword.value, ast.Name)
        result_names.append(keyword.value.id)

    assert result_names == [
        "episode_result_now",  # Disconnect
        "episode_result",      # Early
        "ue_episode_result",   # Normal
    ]
