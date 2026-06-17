import time
import numpy as np
from observation_builder import ACTION_NAME_TO_ID
from constant import ROLLOUT_DIR

def append_rollout_step(buffer, frames, extra, logits, prob, 
                        proposed_action, final_action, info, 
                        pol_state, frame_id_end, fire_frame,
                        ue_attack_active, ue_attack_start, ue_attack_end, 
                        ue_boss_hit, ue_player_hit, ue_episode_done,
                        reward, done):
    step = {
        "frames": frames.squeeze(0).detach().cpu().numpy(),   # shape (C,T,H,W)
        "extra": extra.squeeze(0).detach().cpu().numpy(),     # shape (24,)
        "logits": logits.squeeze(0).detach().cpu().numpy(),   # shape
        "probs": prob.squeeze(0).detach().cpu().numpy(),       # shape (num_actions,)
        "proposed_action_id": np.int64(ACTION_NAME_TO_ID[proposed_action]),
        "final_action_id": np.int64(ACTION_NAME_TO_ID[final_action]),
        "frame_id_end": np.int64(frame_id_end),
        "fire_frame": np.int64(-1 if fire_frame is None else fire_frame),
        "hold_until_frame": np.int64(pol_state["hold_until_frame"]),
        "visible": np.int64(info["visible"]),
        "phase": info["phase"],
        "search_hint": info["search_hint"] if info["search_hint"] is not None else "none",
        "motion": np.float32(info["motion"]),
        "reward": np.float32(reward),
        "done": np.int64(done),
        "ue_attack_active": np.int64(1 if ue_attack_active else 0),
        "ue_attack_start": np.int64(1 if ue_attack_start else 0),
        "ue_attack_end": np.int64(1 if ue_attack_end else 0),
        "ue_boss_hit": np.int64(1 if ue_boss_hit else 0),
        "ue_player_hit": np.int64(1 if ue_player_hit else 0),
        "ue_episode_done": np.int64(1 if ue_episode_done else 0),
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
    out_path = out_dir / f"rollout_{timestamp}.npz"

    np.savez(
        out_path,
        frames=np.stack([x["frames"] for x in buffer], axis=0),                  # (N,3,8,192,192)
        extra=np.stack([x["extra"] for x in buffer], axis=0),                    # (N,24)
        logits=np.stack([x["logits"] for x in buffer], axis=0),                  # (N,10)
        probs=np.stack([x["probs"] for x in buffer], axis=0),                    # (N,10)
        proposed_action_id=np.asarray([x["proposed_action_id"] for x in buffer]),
        final_action_id=np.asarray([x["final_action_id"] for x in buffer]),
        frame_id_end=np.asarray([x["frame_id_end"] for x in buffer]),
        fire_frame=np.asarray([x["fire_frame"] for x in buffer]),
        hold_until_frame=np.asarray([x["hold_until_frame"] for x in buffer]),
        visible=np.asarray([x["visible"] for x in buffer]),
        phase=np.asarray([x["phase"] for x in buffer]),
        search_hint=np.asarray([x["search_hint"] for x in buffer]),
        motion=np.asarray([x["motion"] for x in buffer], dtype=np.float32),
        reward=np.asarray([x["reward"] for x in buffer], dtype=np.float32),
        done=np.asarray([x["done"] for x in buffer], dtype=np.int64),
        ue_attack_active=np.asarray([x["ue_attack_active"] for x in buffer], dtype=np.int64),
        ue_attack_start=np.asarray([x["ue_attack_start"] for x in buffer], dtype=np.int64),
        ue_attack_end=np.asarray([x["ue_attack_end"] for x in buffer], dtype=np.int64),
        ue_boss_hit=np.asarray([x["ue_boss_hit"] for x in buffer], dtype=np.int64),
        ue_player_hit=np.asarray([x["ue_player_hit"] for x in buffer], dtype=np.int64),
        ue_episode_done=np.asarray([x["ue_episode_done"] for x in buffer], dtype=np.int64),
        # value=np.asarray([x["value"] for x in buffer], dtype=np.float32),
    )

    print(f"[rollout] saved {len(buffer)} steps -> {out_path}")
    buffer.clear()

def append_last_step(
    rollout_buffer,
    last_step_cache,
):
    if last_step_cache is None:
        return False

    info = last_step_cache["info"]
    final_action = last_step_cache["final_action"]

    terminal_reward = compute_shaping_reward(
        info=info,
        final_action=final_action,
        ue_attack_active=False,
        ue_attack_start=False,
        ue_boss_hit=False,
        ue_player_hit=False,
        ue_episode_done=True,
    )

    append_rollout_step(
        rollout_buffer,
        last_step_cache["frames"],
        last_step_cache["extra"],
        last_step_cache["logits"],
        last_step_cache["probs"],
        last_step_cache["proposed_action"],
        last_step_cache["final_action"],
        last_step_cache["info"],
        last_step_cache["pol_state"],
        last_step_cache["frame_id_end"],
        last_step_cache["fire_frame"],
        False,   # ue_attack_active
        False,   # ue_attack_start
        False,   # ue_attack_end
        False,   # ue_boss_hit
        False,   # ue_player_hit
        True,    # ue_episode_done
        terminal_reward,
        1,
    )
    return True


def compute_shaping_reward(
        info, final_action, 
        ue_attack_active, ue_attack_start,
        ue_boss_hit, ue_player_hit,
        ue_episode_done
    ):
    reward = 0.0

    visible = info.get("visible", 0)
    phase = info.get("phase", "patrol")

    # 1) 玩家攻擊起手時，Boss 做 evasive 給較大正分
    if ue_attack_start:
        if final_action in {"EvadeBack", "Retreat"}:
            reward += 1.0
        else:
            reward -= 0.2

    # 2) 玩家持續攻擊時，Boss 若仍維持 evasive 類行為，給小正分
    if ue_attack_active:
        if final_action in {"EvadeBack", "Retreat"}:
            reward += 0.2

    # 3) track 時不要一直 Hold
    if visible == 1 and phase == "track":
        if final_action == "Hold":
            reward -= 0.1
        elif final_action in {"Advance", "StrafeLeft", "StrafeRight"}:
            reward += 0.1

    # 4) reacq 時做 SearchTurn 給小正分
    if phase == "reacq":
        if final_action in {"SearchTurnLeft", "SearchTurnRight"}:
            reward += 0.1

    # 5) patrol 時做 PatrolStep 給小正分
    if phase == "patrol":
        if final_action in {"PatrolStepLeft", "PatrolStepRight"}:
            reward += 0.05
    
    # 6) Boss 被玩家打中：負分
    if ue_boss_hit:
        reward -= 1.0

    # 7) 玩家被 Boss 打中：正分
    if ue_player_hit:
        reward += 1.0

    # 8) 回合結束：先給一個小終局 shaping
    # 目前沒有勝負資訊，所以先只做輕微處理
    if ue_episode_done:
        reward += 0.0
        print("happend")

    return np.float32(reward)