from dataclasses import dataclass, field
from collections import Counter
from time import monotonic

@dataclass
class EpisodeEventAudit:
    player_hits_on_boss_raw: int = 0
    boss_hits_on_player_raw: int = 0

    def reset(self):
        self.player_hits_on_boss_raw = 0
        self.boss_hits_on_player_raw = 0

@dataclass
class EpisodeMetrics:
    actor_id: str

    episode_steps: int = 0
    policy_step: int | None = None

    selected_return: float = 0.0

    high_reward_total: float = 0.0
    medium_reward_total: float = 0.0
    low_reward_total: float = 0.0

    boss_hits_on_player: int = 0
    player_hits_on_boss: int = 0
    raw_player_hits_on_boss: int = 0
    raw_boss_hits_on_player: int = 0

    action_counts: Counter = field(default_factory=Counter)

    episode_result: int = 0
    start_time: float = field(default_factory=monotonic)
    duration: float = 0.0

    def record_step(
        self,
        *,
        final_action: str,
        rewards: dict,
        selected_reward: float,
        boss_hits_on_player: int,
        player_hits_on_boss: int,
    ):
        self.episode_steps += 1
        self.selected_return += selected_reward

        self.high_reward_total += rewards.get("high_reward", 0.0)
        self.medium_reward_total += rewards.get("medium_reward", 0.0)
        self.low_reward_total += rewards.get("low_reward", 0.0)

        self.boss_hits_on_player += boss_hits_on_player
        self.player_hits_on_boss += player_hits_on_boss

        self.action_counts[final_action] += 1

    def record_transition(
        self,
        *,
        cache: dict,
        rewards: dict,
        selected_reward: float,
    ):
        self.record_step(
            final_action=cache["final_action"],
            rewards=rewards,
            selected_reward=selected_reward,
            boss_hits_on_player=cache["ue_player_hit_count"],
            player_hits_on_boss=cache["ue_boss_hit_count"],
        )

    def finish_episode(self, episode_result: int, event_audit, policy_step: int | None = None) -> dict:
        self.episode_result = episode_result
        self.policy_step = policy_step
        self.duration = monotonic() - self.start_time

        self.raw_player_hits_on_boss = (event_audit["player_hits_on_boss_raw"])
        self.raw_boss_hits_on_player = (event_audit["boss_hits_on_player_raw"])

        return self.build_summary()

    def build_summary(self):
        if self.episode_result == 1:
            result = "Boss Win"
        elif self.episode_result == -1:
            result = "Boss Loss"
        else:
            result = "Unknown"

        return {
            "actor_id": self.actor_id,
            "policy_step": self.policy_step,

            "result": result,
            "steps": self.episode_steps,
            "duration_sec": self.duration,

            "selected_return": self.selected_return,

            "high_reward_total": self.high_reward_total,
            "medium_reward_total": self.medium_reward_total,
            "low_reward_total": self.low_reward_total,

            "boss_hits_on_player": self.boss_hits_on_player,
            "player_hits_on_boss": self.player_hits_on_boss,
            "raw_player_hits_on_boss": self.raw_player_hits_on_boss,
            "raw_boss_hits_on_player": self.raw_boss_hits_on_player,

            "action_counts": dict(self.action_counts),
        }

def format_summary(summary: dict) -> str:
    return (
        "========== Episode Summary ==========\n"
        f"Actor {summary['actor_id']} | "
        f"Policy Step: {summary['policy_step']} | \n"
        f"Result: {summary['result']} | "
        f"Steps: {summary['steps']} | "
        f"Duration: {summary['duration_sec']:.2f} sec | \n"
        f"Selected Return: {summary['selected_return']:+.2f} | \n"
        "\nReward Channels:\n"
        f"  High: {summary['high_reward_total']:+.2f} | "
        f"  Medium: {summary['medium_reward_total']:+.2f} | "
        f"  Low: {summary['low_reward_total']:+.2f} | \n"
        "\nActions:\n"
        f"  Advance: {summary['action_counts'].get('Advance', 0)}\n"
        f"  Hold: {summary['action_counts'].get('Hold', 0)}\n"
        f"  StrafeLeft: {summary['action_counts'].get('StrafeLeft', 0)}\n"
        f"  StrafeRight: {summary['action_counts'].get('StrafeRight', 0)}\n"
        f"  SearchTurnLeft: {summary['action_counts'].get('SearchTurnLeft', 0)}\n"
        f"  SearchTurnRight: {summary['action_counts'].get('SearchTurnRight', 0)}\n"
        f"  EvadeBack: {summary['action_counts'].get('EvadeBack', 0)}\n"
        f"  PatrolRight: {summary['action_counts'].get('PatrolStepRight', 0)}\n"
        f"  PatrolLeft: {summary['action_counts'].get('PatrolStepLeft', 0)}\n"
        "====================================="

        "\nAttributed Hits:\n"
        f"  Boss -> Player: {summary['boss_hits_on_player']} | "
        f"  Player -> Boss: {summary['player_hits_on_boss']} | \n"

        "\nRaw UE Hit Events:\n"
        f"  Boss -> Player: {summary['raw_boss_hits_on_player']} | "
        f"  Player -> Boss: {summary['raw_player_hits_on_boss']} |\n"
        "====================================="
    )