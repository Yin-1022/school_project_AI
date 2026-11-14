"""Regression tests for Step 4-2C-3A TransitionTracker extraction.

Expected module: scripts/transition_tracker.py
Run from repo root:
    $env:PYTHONPATH = "scripts"       # PowerShell
    python -m pytest scripts/tests/test_transition_tracker.py -v
"""

import numpy as np
import pytest
import torch

import transition_tracker as tracker_module
from constant import BOSS_MAX_HP, PLAYER_MAX_HP
from episode_metrics import EpisodeEventAudit
from observation_builder import ACTION_NAME_TO_ID
from transition_tracker import TransitionTracker


def make_policy_cache(action="Advance", frame_id=8):
    """A minimal valid Policy cache accepted by the real rollout_logger."""
    num_actions = len(ACTION_NAME_TO_ID)
    probs = torch.full((1, num_actions), 1.0 / num_actions)

    return {
        "frames": torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32),
        "extra": torch.zeros((1, 24), dtype=torch.float32),
        "logits": torch.zeros((1, num_actions), dtype=torch.float32),
        "probs": probs,
        "behavior_probs": np.full(num_actions, 1.0 / num_actions, dtype=np.float32),
        "proposed_action": action,
        "final_action": action,
        "info": {"visible": 1, "phase": "track", "search_hint": "center", "motion": 0.0},
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
    """Exercise real step/reward conversion, but never write rollout files."""
    audit = EpisodeEventAudit(player_max_hp=PLAYER_MAX_HP, boss_max_hp=BOSS_MAX_HP)
    tracker = TransitionTracker(actor_id="actor_test", event_audit=audit)
    saved = []

    def fake_flush(buffer, actor_id, bootstrap_cathe=None):
        saved.append({
            "actor_id": actor_id,
            "steps": list(buffer),
            "bootstrap_cache": bootstrap_cathe,
        })
        buffer.clear()

    monkeypatch.setattr(tracker_module, "flush_rollout_buffer", fake_flush)
    monkeypatch.setattr(tracker_module, "ROLLOUT_SAVE_EVERY", 999)
    return tracker, audit, saved


def record_events(tracker, **updates):
    params = {
        "boss_hit_count": 0,
        "player_hit_count": 0,
        "boss_damage": 0.0,
        "player_damage": 0.0,
        "att1_start": False,
        "att1_end": False,
        "att2_start": False,
        "att2_end": False,
        "episode_done": False,
        "episode_result": 0,
    }
    params.update(updates)
    tracker.record_frame_events(**params)


def test_two_policy_actions_rotate_and_terminal_reward(env):
    tracker, audit, saved = env

    tracker.record_policy_sent(make_policy_cache("Advance", 8))
    record_events(tracker, player_hit_count=1, player_damage=10.0, att1_start=True)
    tracker.rotate_at_decision_boundary()

    tracker.record_policy_sent(make_policy_cache("StrafeRight", 12))
    tracker.finalize_transitions(episode_result=1)
    tracker.flush()

    assert len(saved) == 1
    first, last = saved[0]["steps"]
    assert [int(first["done"]), int(last["done"])] == [0, 1]
    assert [int(first["final_action_id"]), int(last["final_action_id"])] == [
        ACTION_NAME_TO_ID["Advance"], ACTION_NAME_TO_ID["StrafeRight"]
    ]
    assert float(first["reward_medium"]) == pytest.approx(10.0 / PLAYER_MAX_HP)
    assert int(first["ue_att1_start"]) == 1
    assert float(first["reward_high"]) == pytest.approx(0.0)
    assert float(last["reward_high"]) == pytest.approx(5.0)
    assert tracker.episode_metrics.episode_steps == 2
    assert tracker.episode_metrics.selected_return == pytest.approx(
        10.0 / PLAYER_MAX_HP + 5.0
    )
    assert audit.unattributed_boss_hits_on_player == 0


def test_rejected_policy_dropped_and_combat_becomes_unattributed(env, capsys):
    tracker, audit, saved = env
    tracker.record_policy_sent(make_policy_cache("StrafeLeft"))
    record_events(
        tracker,
        boss_hit_count=1,
        player_hit_count=2,
        boss_damage=30.0,
        player_damage=5.0,
    )

    tracker.discard_rejected_policy()
    tracker.finalize_transitions(episode_result=-1)
    tracker.flush()

    assert saved == []  # Rejected step cannot receive a fabricated terminal reward.
    assert tracker.episode_metrics.episode_steps == 0
    assert audit.unattributed_player_hits_on_boss == 1
    assert audit.unattributed_boss_hits_on_player == 2
    assert audit.unattributed_player_damage_to_boss == pytest.approx(30.0)
    assert audit.unattributed_boss_damage_to_player == pytest.approx(5.0)
    assert "without any trainable policy transition" in capsys.readouterr().out


def test_rejection_preserves_previous_valid_anchor_for_terminal(env):
    tracker, audit, saved = env

    tracker.record_policy_sent(make_policy_cache("Advance", 8))
    tracker.rotate_at_decision_boundary()
    tracker.record_policy_sent(make_policy_cache("StrafeLeft", 12))
    record_events(tracker, boss_hit_count=1, boss_damage=20.0)
    tracker.discard_rejected_policy()

    tracker.finalize_transitions(episode_result=-1)
    tracker.flush()

    assert len(saved) == 1
    assert len(saved[0]["steps"]) == 1
    step = saved[0]["steps"][0]
    assert int(step["final_action_id"]) == ACTION_NAME_TO_ID["Advance"]
    assert int(step["done"]) == 1
    assert float(step["reward_high"]) == pytest.approx(-5.0)
    assert float(step["reward_medium"]) == pytest.approx(0.0)
    assert int(step["ue_boss_hit_count"]) == 0
    assert audit.unattributed_player_damage_to_boss == pytest.approx(20.0)


def test_runtime_interval_combat_unattributed_but_anchor_retained(env):
    tracker, audit, saved = env

    tracker.record_policy_sent(make_policy_cache("Advance"))
    tracker.rotate_at_decision_boundary()  # No active Policy cache during Runtime action.
    record_events(tracker, player_hit_count=1, player_damage=15.0)

    tracker.finalize_transitions(episode_result=1)
    tracker.flush()

    assert len(saved) == 1
    step = saved[0]["steps"][0]
    assert int(step["done"]) == 1
    assert float(step["reward_high"]) == pytest.approx(5.0)
    assert float(step["reward_medium"]) == pytest.approx(0.0)
    assert int(step["ue_player_hit_count"]) == 0
    assert audit.unattributed_boss_hits_on_player == 1
    assert audit.unattributed_boss_damage_to_player == pytest.approx(15.0)


def test_terminal_pending_combat_attributed_once_to_active_policy(env):
    tracker, audit, saved = env
    tracker.record_policy_sent(make_policy_cache("Advance"))

    tracker.record_terminal_combat({
        "pending_boss_hits": 1,
        "pending_player_hits": 2,
        "pending_boss_damage": 25.0,
        "pending_player_damage": 10.0,
    })
    tracker.finalize_transitions(episode_result=1)
    tracker.flush()

    step = saved[0]["steps"][0]
    assert int(step["done"]) == 1
    assert int(step["ue_boss_hit_count"]) == 1
    assert int(step["ue_player_hit_count"]) == 2
    assert float(step["ue_boss_damage"]) == pytest.approx(25.0)
    assert float(step["ue_player_damage"]) == pytest.approx(10.0)
    assert float(step["reward_medium"]) == pytest.approx(
        10.0 / PLAYER_MAX_HP - 25.0 / BOSS_MAX_HP
    )
    assert float(step["reward_high"]) == pytest.approx(5.0)
    assert audit.unattributed_player_damage_to_boss == 0.0
    assert tracker.episode_metrics.episode_steps == 1


def test_auto_flush_uses_new_anchor_as_bootstrap(env, monkeypatch):
    tracker, _, saved = env
    monkeypatch.setattr(tracker_module, "ROLLOUT_SAVE_EVERY", 1)

    tracker.record_policy_sent(make_policy_cache("Advance", 8))
    tracker.rotate_at_decision_boundary()
    next_cache = make_policy_cache("StrafeRight", 12)
    tracker.record_policy_sent(next_cache)
    tracker.rotate_at_decision_boundary()  # Appends A and flushes with B as bootstrap.

    assert len(saved) == 1
    assert len(saved[0]["steps"]) == 1
    assert int(saved[0]["steps"][0]["done"]) == 0
    assert saved[0]["bootstrap_cache"] is next_cache

    tracker.finalize_transitions(episode_result=1)
    tracker.flush()
    assert len(saved) == 2
    assert len(saved[1]["steps"]) == 1
    assert int(saved[1]["steps"][0]["final_action_id"]) == ACTION_NAME_TO_ID["StrafeRight"]
    assert int(saved[1]["steps"][0]["done"]) == 1


def test_reset_episode_preserves_rollout_buffer_as_legacy_behavior(env):
    tracker, _, saved = env
    tracker.record_policy_sent(make_policy_cache("Advance"))
    tracker.finalize_transitions(episode_result=1)  # Buffered, not flushed yet.
    assert tracker.episode_metrics.episode_steps == 1

    tracker.reset_episode_state()
    assert tracker.episode_metrics.episode_steps == 0

    tracker.flush()
    assert len(saved) == 1
    assert len(saved[0]["steps"]) == 1
    assert int(saved[0]["steps"][0]["done"]) == 1


def test_disconnect_without_terminal_drops_buffer(env):
    tracker, _, saved = env
    tracker.record_policy_sent(make_policy_cache("Advance", 8))
    tracker.rotate_at_decision_boundary()
    tracker.record_policy_sent(make_policy_cache("StrafeRight", 12))
    tracker.rotate_at_decision_boundary()  # Buffer contains A, done=0.

    tracker.drop_buffer_on_disconnect()
    tracker.reset_episode_state()
    tracker.flush()

    assert saved == []
    assert tracker.episode_metrics.episode_steps == 0


def test_no_valid_policy_terminal_does_not_invent_step(env, capsys):
    tracker, _, saved = env
    tracker.finalize_transitions(episode_result=1)
    tracker.flush()

    assert saved == []
    assert tracker.episode_metrics.episode_steps == 0
    assert "without any trainable policy transition" in capsys.readouterr().out
