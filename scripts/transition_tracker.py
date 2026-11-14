
from constant import ROLLOUT_SAVE_EVERY
from episode_metrics import (
    EpisodeEventAudit,
    EpisodeMetrics,
    attribute_combat_events,
)
from rollout_logger import append_cached_step, flush_rollout_buffer
from reward_priority import resolve_priority_reward


class TransitionTracker:
    """IMPALA transition state owner."""

    def __init__(self, *, actor_id: str, event_audit: EpisodeEventAudit):
        self.actor_id = actor_id
        self.event_audit = event_audit
        self.episode_metrics = EpisodeMetrics(actor_id=actor_id)

        self._rollout_buffer: list[dict] = []
        self._last_step_cache: dict | None = None
        self._terminal_anchor_cache: dict | None = None

    def clear_pending(self) -> None:
        self._last_step_cache = None
        self._terminal_anchor_cache = None

    def reset_episode_state(self) -> None:
        self.clear_pending()
        self.episode_metrics = EpisodeMetrics(actor_id=self.actor_id)

        # 暫時保留原本行為：Episode Start 不清空 Rollout Buffer

    def drop_buffer_on_disconnect(self) -> None:
        if self._rollout_buffer:
            print(
                f"[rollout] dropping {len(self._rollout_buffer)} steps "
                "on disconnect without terminal result"
            )
            self._rollout_buffer.clear()

    def record_policy_sent(self, cache: dict) -> None:
        self._last_step_cache = cache

    def record_frame_events(
        self,
        *,
        boss_hit_count: int,
        player_hit_count: int,
        boss_damage: float,
        player_damage: float,
        att1_start: bool,
        att1_end: bool,
        att2_start: bool,
        att2_end: bool,
        episode_done: bool,
        episode_result: int,
    ) -> None:
        cache = self._last_step_cache

        attribute_combat_events(
            last_step_cache=cache,
            event_audit=self.event_audit,
            boss_hit_count=boss_hit_count,
            player_hit_count=player_hit_count,
            boss_damage=boss_damage,
            player_damage=player_damage,
        )

        if cache is None:
            return

        if att1_start:
            cache["ue_att1_start"] = True
        if att1_end:
            cache["ue_att1_end"] = True
        if att2_start:
            cache["ue_att2_start"] = True
        if att2_end:
            cache["ue_att2_end"] = True

        if episode_done:
            cache["ue_episode_done"] = True
            cache["ue_episode_result"] = episode_result

    def record_terminal_combat(self, snapshot: dict) -> None:
        attribute_combat_events(
            last_step_cache=self._last_step_cache,
            event_audit=self.event_audit,
            boss_hit_count=snapshot["pending_boss_hits"],
            player_hit_count=snapshot["pending_player_hits"],
            boss_damage=snapshot["pending_boss_damage"],
            player_damage=snapshot["pending_player_damage"],
        )

    def discard_rejected_policy(self) -> None:
        cache = self._last_step_cache
        if cache is None:
            return

        attribute_combat_events(
            last_step_cache=None,
            event_audit=self.event_audit,
            boss_hit_count=cache["ue_boss_hit_count"],
            player_hit_count=cache["ue_player_hit_count"],
            boss_damage=cache["ue_boss_damage"],
            player_damage=cache["ue_player_damage"],
        )

        self._last_step_cache = None

    def rotate_at_decision_boundary(self) -> None:
        if self._last_step_cache is None:
            return

        if self._terminal_anchor_cache is not None:
            self._append_and_record_step(
                self._terminal_anchor_cache,
                done=0,
            )

        self._terminal_anchor_cache = self._last_step_cache
        self._last_step_cache = None

        if len(self._rollout_buffer) >= ROLLOUT_SAVE_EVERY:
            flush_rollout_buffer(
                self._rollout_buffer,
                self.actor_id,
                self._terminal_anchor_cache,
            )

    def finalize_transitions(self, episode_result: int) -> None:
        if self._last_step_cache is not None:
            if self._terminal_anchor_cache is not None:
                self._append_and_record_step(
                    self._terminal_anchor_cache,
                    done=0,
                )

            cache = self._last_step_cache

        elif self._terminal_anchor_cache is not None:
            cache = self._terminal_anchor_cache

        else:
            cache = None
            print(
                "[episode warning] terminal received "
                "without any trainable policy transition"
            )

        if cache is not None:
            cache["ue_episode_done"] = True
            cache["ue_episode_result"] = episode_result

            self._append_and_record_step(cache, done=1)

        self.clear_pending()

    def _append_and_record_step(
        self,
        cache: dict,
        *,
        done: int,
    ) -> None:
        rewards = append_cached_step(
            self._rollout_buffer,
            cache,
            done=done,
        )

        if rewards is None:
            return

        resolved = resolve_priority_reward(
            reward_high=float(rewards["high_reward"]),
            reward_medium=float(rewards["medium_reward"]),
            reward_low=float(rewards["low_reward"]),
            ue_player_damage=float(cache["ue_player_damage"]),
            ue_boss_damage=float(cache["ue_boss_damage"]),
            ue_episode_result=cache["ue_episode_result"],
        )

        self.episode_metrics.record_transition(
            cache=cache,
            rewards=rewards,
            selected_reward=resolved["reward"],
        )

    def flush(self) -> None:
        if self._rollout_buffer:
            flush_rollout_buffer(
                self._rollout_buffer,
                self.actor_id,
            )
