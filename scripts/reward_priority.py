PRIORITY_NONE = 0
PRIORITY_LOW = 1
PRIORITY_MEDIUM = 2
PRIORITY_HIGH = 3

def resolve_priority_reward(
    reward_high: float,
    reward_medium: float,
    reward_low: float,
    ue_player_damage: float,
    ue_boss_damage: float,
    ue_episode_result: int,
) -> dict:
    high_active = ue_episode_result != 0

    medium_active = (
        ue_player_damage > 1e-8
        or ue_boss_damage > 1e-8
    )

    low_active = abs(reward_low) > 1e-8

    if high_active:
        return {
            "priority": PRIORITY_HIGH,
            "reward": reward_high,
        }

    if medium_active:
        return {
            "priority": PRIORITY_MEDIUM,
            "reward": reward_medium,
        }

    if low_active:
        return {
            "priority": PRIORITY_LOW,
            "reward": reward_low,
        }

    return {
        "priority": PRIORITY_NONE,
        "reward": 0.0,
    }