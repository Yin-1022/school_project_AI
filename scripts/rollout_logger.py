import time
import numpy as np
from observation_builder import ACTION_NAME_TO_ID
from constant import ROLLOUT_DIR

def append_rollout_step(buffer, frames, extra, logits, probs, behavior_probs,
                        proposed_action, final_action, info, 
                        pol_state, frame_id_end, fire_frame,
                        ue_att1_start, ue_att1_end,
                        ue_att2_start, ue_att2_end,
                        ue_boss_hit_count, ue_player_hit_count, ue_episode_done,
                        reward_high, reward_medium, reward_low, done, action_mask=None):
    step = {
        "frames": frames.squeeze(0).detach().cpu().numpy(),   # shape (C,T,H,W)
        "extra": extra.squeeze(0).detach().cpu().numpy(),     # shape (24,)
        "logits": logits.squeeze(0).detach().cpu().numpy(),   # shape
        "probs": probs.squeeze(0).detach().cpu().numpy(),     # shape (num_actions,)
        "behavior_probs": np.asarray(behavior_probs, dtype=np.float32),
        "proposed_action_id": np.int64(ACTION_NAME_TO_ID[proposed_action]),
        "final_action_id": np.int64(ACTION_NAME_TO_ID[final_action]),
        "frame_id_end": np.int64(frame_id_end),
        "fire_frame": np.int64(-1 if fire_frame is None else fire_frame),
        "hold_until_frame": np.int64(pol_state["hold_until_frame"]),
        "visible": np.int64(info["visible"]),
        "phase": info["phase"],
        "search_hint": info["search_hint"] if info["search_hint"] is not None else "none",
        "motion": np.float32(info["motion"]),
        "reward_high": np.float32(reward_high),
        "reward_medium": np.float32(reward_medium),
        "reward_low": np.float32(reward_low),
        "done": np.int64(done),
        "ue_att1_start": np.int64(1 if ue_att1_start else 0),
        "ue_att1_end": np.int64(1 if ue_att1_end else 0),
        "ue_att2_start": np.int64(1 if ue_att2_start else 0),
        "ue_att2_end": np.int64(1 if ue_att2_end else 0),
        "ue_boss_hit_count": np.int64(ue_boss_hit_count),
        "ue_player_hit_count": np.int64(ue_player_hit_count),
        "ue_episode_done": np.int64(1 if ue_episode_done else 0),
        "action_mask": np.asarray(action_mask, dtype=np.bool_),
        # "value": np.float32(
        #         value.detach().cpu().item() if hasattr(value, "detach") else value
        #     ),
    }
    buffer.append(step)

def flush_rollout_buffer(buffer):
    if not buffer:
        return
    
    out_dir = ROLLOUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    out_path = out_dir / f"rollout_v2_{timestamp}.npz"

    np.savez(
        out_path,

        frames=np.stack([x["frames"] for x in buffer],axis=0),
        extra=np.stack([x["extra"] for x in buffer],axis=0),
        logits=np.stack([x["logits"] for x in buffer],axis=0),
        probs=np.stack([x["probs"] for x in buffer],axis=0),
        behavior_probs=np.stack([x["behavior_probs"] for x in buffer],axis=0),
        proposed_action_id=np.asarray([x["proposed_action_id"] for x in buffer]),
        final_action_id=np.asarray([x["final_action_id"] for x in buffer]),
        frame_id_end=np.asarray([x["frame_id_end"] for x in buffer]),
        fire_frame=np.asarray([x["fire_frame"] for x in buffer]),
        hold_until_frame=np.asarray([x["hold_until_frame"] for x in buffer]),
        visible=np.asarray([x["visible"] for x in buffer]),
        phase=np.asarray([x["phase"] for x in buffer]),
        search_hint=np.asarray([x["search_hint"] for x in buffer]),
        motion=np.asarray([x["motion"] for x in buffer],dtype=np.float32),

        reward_high=np.asarray([x["reward_high"] for x in buffer],dtype=np.float32),
        reward_medium=np.asarray([x["reward_medium"] for x in buffer],dtype=np.float32),
        reward_low=np.asarray([x["reward_low"] for x in buffer],dtype=np.float32),

        done=np.asarray([x["done"] for x in buffer],dtype=np.int64),
        ue_att1_start=np.asarray([x["ue_att1_start"] for x in buffer],dtype=np.int64),
        ue_att1_end=np.asarray([x["ue_att1_end"] for x in buffer],dtype=np.int64),
        ue_att2_start=np.asarray([x["ue_att2_start"] for x in buffer],dtype=np.int64),
        ue_att2_end=np.asarray([x["ue_att2_end"] for x in buffer],dtype=np.int64),
        ue_boss_hit_count=np.asarray([x["ue_boss_hit_count"] for x in buffer],dtype=np.int64),
        ue_player_hit_count=np.asarray([x["ue_player_hit_count"] for x in buffer],dtype=np.int64),
        ue_episode_done=np.asarray([x["ue_episode_done"] for x in buffer],dtype=np.int64),

        action_mask=np.stack([x["action_mask"] for x in buffer],axis=0),
    )

    print(f"[rollout] saved {len(buffer)} steps -> {out_path}")
    buffer.clear()

def append_last_step(
    rollout_buffer,
    last_step_cache,
):
    if last_step_cache is None:
        return False

    last_step_cache["ue_episode_done"] = True

    append_cached_step(
        rollout_buffer,
        last_step_cache,
        done=1,
    )

    return True

def compute_reward_channels(
        info,final_action,
        ue_att1_start, ue_att1_end, ue_att2_start, ue_att2_end,
        ue_boss_hit_count, ue_player_hit_count,
    ):

    high_reward = 0.0
    medium_reward = 0.0
    low_reward = 0.0

    #High level shaping reward

    #Medium level shaping reward
    medium_reward += ue_player_hit_count * 1.0
    medium_reward -= ue_boss_hit_count * 1.0

    #Low level shaping reward
    visible = info.get("visible", 0)
    phase = info.get("phase", "patrol")

    # # 1) 玩家攻擊起手時，Boss 做 evasive 給較大正分
    # if ue_attack_start and final_action in {"EvadeBack", "Retreat"}:
    #         low_reward += 1.0

    # 2) track 時不要一直 Hold
    if visible == 1 and phase == "track":
        if final_action == "Hold":
            low_reward -= 0.1
        elif final_action in {"Advance", "StrafeLeft", "StrafeRight"}:
            low_reward += 0.1

    # 3) reacq 時做 SearchTurn 給小正分
    if phase == "reacq":
        if final_action in {"SearchTurnLeft", "SearchTurnRight"}:
            low_reward += 0.1

    # 4) patrol 時做 PatrolStep 給小正分
    if phase == "patrol":
        if final_action in {"PatrolStepLeft", "PatrolStepRight"}:
            low_reward += 0.05

    return {
        "high_reward": np.float32(high_reward),
        "medium_reward": np.float32(medium_reward),
        "low_reward": np.float32(low_reward)
    }

def append_cached_step(rollout_buffer, cache, done=0):
    if cache is None:
        return False

    rewards = compute_reward_channels(
        info=cache["info"],
        final_action=cache["final_action"],
        ue_att1_start=cache["ue_att1_start"],
        ue_att1_end=cache["ue_att1_end"],
        ue_att2_start=cache["ue_att2_start"],
        ue_att2_end=cache["ue_att2_end"],
        ue_boss_hit_count=cache["ue_boss_hit_count"],
        ue_player_hit_count=cache["ue_player_hit_count"],
    )

    append_rollout_step(
        buffer=rollout_buffer,
        frames=cache["frames"],
        extra=cache["extra"],
        logits=cache["logits"],
        probs=cache["probs"],
        behavior_probs=cache["behavior_probs"],
        proposed_action=cache["proposed_action"],
        final_action=cache["final_action"],
        info=cache["info"],
        pol_state=cache["pol_state"],
        frame_id_end=cache["frame_id_end"],
        fire_frame=cache["fire_frame"],

        ue_att1_start=cache["ue_att1_start"],
        ue_att1_end=cache["ue_att1_end"],
        ue_att2_start=cache["ue_att2_start"],
        ue_att2_end=cache["ue_att2_end"],
        ue_boss_hit_count=cache["ue_boss_hit_count"],
        ue_player_hit_count=cache["ue_player_hit_count"],
        ue_episode_done=cache["ue_episode_done"],

        reward_high=rewards["high_reward"],
        reward_medium=rewards["medium_reward"],
        reward_low=rewards["low_reward"],

        done=done,
        action_mask=cache["action_mask"],
    )

    return True