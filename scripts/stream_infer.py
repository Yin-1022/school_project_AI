import cv2
from pathlib import Path
import collections
import torch
import numpy as np
from visibility import update as vis_update
from policy import init_state as policy_init, step as rule_policy_step
from stream_io import send_action, receive_from_ue, tcp_frame_stream
from observation_builder import build_frame_tensor, build_extra_tensor, ACTION_NAME_TO_ID
from policy_inference import load_model, infer_action
from action_postprocess import apply_action_with_state
from rollout_logger import append_rollout_step, flush_rollout_buffer, compute_shaping_reward, append_last_step
import threading
import time
from constant import (
    ACTION_ID_TO_NAME,
    ROLLOUT_SAVE_EVERY,
    POLICY_MODE,
)

# RAW_DIR = Path("data/raw_videos")
# video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
SEQ = 0
UE_EVENT_STATE = {
    "attack_active": False,
    "attack_start_pulse": False,
    "attack_end_pulse": False,
    "boss_hit_pulse": False,
    "player_hit_pulse": False,
    "episode_done_flag": False,
}
UE_EVENT_LOCK = threading.Lock()

def main():
    rollout_buffer = []
    last_step_cache = None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = None
        if POLICY_MODE == "bc":
            model = load_model(str(WEIGHTS_PATH), device=device)
        receive_from_ue(UE_EVENT_LOCK, UE_EVENT_STATE)

        vis_state = None                    
        pol_state = policy_init() 

        frame_ring_buffer = collections.deque(maxlen=CLIP_FRAMES)
        idx = 0
        pushed_frames = 0
        recv_frames = 0
        sample_every = 1
        global SEQ

        for frame in tcp_frame_stream(host='127.0.0.1', port=9999, img_w=192, img_h=192, img_c=3, debug_show=False):
            if frame is None:
                episode_done_now = False

                # 給 OSC callback 一點時間把 game_over 寫進 shared state
                for _ in range(10):   # 最多等 10 * 0.02 = 0.2 秒
                    with UE_EVENT_LOCK:
                        episode_done_now = UE_EVENT_STATE["episode_done_flag"]
                    if episode_done_now:
                        break
                    time.sleep(0.02)

                if episode_done_now:
                    print("[UE event] episode done (disconnect terminal append)")

                    appended = append_last_step(
                        rollout_buffer=rollout_buffer,
                        last_step_cache=last_step_cache,
                    )

                    if appended:
                        print("[rollout] terminal step appended on disconnect")

                    with UE_EVENT_LOCK:
                        UE_EVENT_STATE["episode_done_flag"] = False

                if rollout_buffer:
                    flush_rollout_buffer(rollout_buffer)

                continue
            
            recv_frames += 1

            if recv_frames % sample_every != 0:
                continue

            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            frame_ring_buffer.append(frame)
            pushed_frames += 1

            if len(frame_ring_buffer) < CLIP_FRAMES:
                continue
            if pushed_frames % CLIP_STRIDE != 0:
                continue

            frame_id_end = pushed_frames - 1
            frames = build_frame_tensor(frame_ring_buffer)

            info, vis_state = vis_update(
                vis_state,
                frames,
                pred_name="idle",
                visible=1,
                frame_id_end=frame_id_end,
            )

            extra_tensor = build_extra_tensor(info, pol_state, frame_id_end)

            with UE_EVENT_LOCK:
                ue_attack_active = UE_EVENT_STATE["attack_active"]
                ue_attack_start = UE_EVENT_STATE["attack_start_pulse"]
                ue_attack_end = UE_EVENT_STATE["attack_end_pulse"]
                ue_boss_hit = UE_EVENT_STATE["boss_hit_pulse"]
                ue_player_hit = UE_EVENT_STATE["player_hit_pulse"]
                ue_episode_done = UE_EVENT_STATE["episode_done_flag"]

                # pulse 讀完就清掉
                UE_EVENT_STATE["attack_start_pulse"] = False
                UE_EVENT_STATE["attack_end_pulse"] = False
                UE_EVENT_STATE["boss_hit_pulse"] = False
                UE_EVENT_STATE["player_hit_pulse"] = False
            
            if ue_attack_start:
                print("[UE event] attack start")
            if ue_attack_end:
                print("[UE event] attack end")
            if ue_boss_hit:
                print("[UE event] boss hit")
            if ue_player_hit:
                print("[UE event] player hit")
            if ue_episode_done:
                print("[UE event] episode done")

            if POLICY_MODE == "bc":
                bc_out = infer_action(frames, extra_tensor, model)
                proposed_action = bc_out["action_name"]
                action_conf = bc_out["conf"]
                topk_actions = [ACTION_ID_TO_NAME[id] for id in bc_out["topk_ids"][0]]
                topk_confs = bc_out["topk_probs"][0]
                logits_for_log = bc_out["logits"]
                probs_for_log = bc_out["probs"]

            elif POLICY_MODE == "rule":
                rule_pred_name, rule_conf = derive_rule_pred_name(info)

                proposed_action, pol_state, _, fire_frame = rule_policy_step(
                    pol_state,
                    pred_name=rule_pred_name,
                    conf=rule_conf,
                    visible=info["visible"],
                    phase=info["phase"],
                    search_hint=info["search_hint"],
                    frame_id_end=frame_id_end,
                )

                action = proposed_action
                action_conf = 1.0
                topk_actions = [proposed_action]
                topk_confs = [1.0]

                # rollout_logger 仍需要 logits/probs，先放假資料
                num_actions = len(ACTION_ID_TO_NAME)
                logits_np = np.zeros((1, num_actions), dtype=np.float32)
                probs_np = np.zeros((1, num_actions), dtype=np.float32)
                action_id = ACTION_NAME_TO_ID[proposed_action]
                logits_np[0, action_id] = 1.0
                probs_np[0, action_id] = 1.0

                logits_for_log = torch.from_numpy(logits_np)
                probs_for_log = torch.from_numpy(probs_np)

            if POLICY_MODE == "bc":
                action, pol_state, fire_frame = apply_action_with_state(
                    pol_state,
                    proposed_action=proposed_action,
                    topk_actions=topk_actions,
                    frame_id_end=frame_id_end,
                    info=info
                )

            reward = compute_shaping_reward(
                info=info,
                final_action=action,
                ue_attack_active=ue_attack_active,
                ue_attack_start=ue_attack_start,
                ue_boss_hit=ue_boss_hit,
                ue_player_hit=ue_player_hit,
                ue_episode_done=ue_episode_done,
            )
            done = 1 if ue_episode_done else 0

            last_step_cache = {
                "frames": frames,
                "extra": extra_tensor,
                "logits": logits_for_log,
                "probs": probs_for_log,
                "proposed_action": proposed_action,
                "final_action": action,
                "info": info,
                "pol_state": dict(pol_state),
                "frame_id_end": frame_id_end,
                "fire_frame": fire_frame,
            }

            append_rollout_step(
                rollout_buffer,
                frames,
                extra_tensor,
                logits_for_log,
                probs_for_log,
                proposed_action,
                action,
                info,
                pol_state,
                frame_id_end,
                fire_frame,
                ue_attack_active,
                ue_attack_start,
                ue_attack_end,
                ue_boss_hit,
                ue_player_hit,
                ue_episode_done,
                reward,
                done,
                # state_value,
            )

            if len(rollout_buffer) >= ROLLOUT_SAVE_EVERY or ue_episode_done:
                flush_rollout_buffer(rollout_buffer)
            
            if ue_episode_done:
                with UE_EVENT_LOCK:
                    UE_EVENT_STATE["episode_done_flag"] = False

            print(
                f"[t={frame_id_end:05d}] "
                f"vis={info['visible']} phase={info['phase']} "
                f"hint={info['search_hint']} motion={info['motion']:.4f} "
                f"→ bc_action={proposed_action}({action_conf:.2f}) "
                f"final_action={action} fire@{fire_frame} hold_until={pol_state['hold_until_frame']}"
                f" topk={list(zip(topk_actions, topk_confs))}"
                # f"value={state_value:.4f}"
            )

            if fire_frame is None:
                continue

            SEQ += 1
            jsonMsg = {
                "type": "boss_action",
                "ts_frame": frame_id_end,
                "fire_frame": fire_frame,
                "hold_until": pol_state["hold_until_frame"],
                "action": action,
                "params": {},
                "meta": {
                    "conf": action_conf,
                    "phase": info["phase"],
                    "search_hint": info["search_hint"],
                },
                "seq": SEQ
            }

            send_action(jsonMsg)
    finally:
         if rollout_buffer:
            flush_rollout_buffer(rollout_buffer)
            print("[rollout] flushed remaining buffer on shutdown")

def derive_rule_pred_name(info):
    if info["visible"] == 0:
        return "none", 1.0

    if info["phase"] == "track":
        return "move", 0.8

    return "idle", 0.6

# def save_teacher_sample(frames, extra, action_name, frame_id_end):
#     out_dir = Path("data/teacher_samples")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     frames = frames.squeeze(0).detach().cpu().numpy()   # shape (C,T,H,W)
#     extra = extra.squeeze(0).detach().cpu().numpy()     # shape (24,)
#     if action_name not in ACTION_NAME_TO_ID:
#         raise ValueError(f"Unknown action_name: {action_name}")
#     action_id = ACTION_NAME_TO_ID[action_name]

#     timestamp = int(time.time() * 1000)
#     out_path = out_dir / f"sample_{timestamp}_{frame_id_end:06d}.npz"

#     np.savez(
#         out_path, 
#         frames=frames, 
#         extra=extra, 
#         action_id=np.int64(action_id),
#         action_name=action_name,
#         frame_id_end=np.int64(frame_id_end),
#     )
#     print(f"Saved teacher sample to {out_path}")
    
if __name__ == "__main__":
    main()