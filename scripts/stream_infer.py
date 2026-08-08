import cv2
from pathlib import Path
import collections
import torch
import numpy as np
from presence_data import save_presence_sample
from visibility import update as vis_update
from policy import init_state as policy_init, step as rule_policy_step
from stream_io import send_action, receive_from_ue, tcp_frame_stream
from observation_builder import build_frame_tensor, build_extra_tensor, ACTION_NAME_TO_ID
from policy_inference import load_model, infer_action, load_action_cls_model, infer_player_state, CLASS_TO_ID
from presence_inference import (
    load_presence_model,
    infer_player_presence,
)
from action_postprocess import apply_action_with_state
from rollout_logger import append_cached_step, flush_rollout_buffer, append_last_step
import threading
import time
from constant import (
    ACTION_ID_TO_NAME,
    ROLLOUT_SAVE_EVERY,
    POLICY_MODE,
    BLOCKING_ACTIONS
)

# RAW_DIR = Path("data/raw_videos")
# video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
ACTION_CLS_WEIGHTS_PATH = Path("data/meta/best_action_cls.pt")
PRESENCE_WEIGHTS_PATH = Path("data/meta/best_presence_avgmax_balanced_hardP.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
SEQ = 0
UE_EVENT_STATE = {
    "att1_active": False,
    "att1_start_pulse": False,
    "att1_end_pulse": False,

    "att2_active": False,
    "att2_start_pulse": False,
    "att2_end_pulse": False,

    "boss_hit_pulse": False,
    "player_hit_pulse": False,
    "episode_done_flag": False,
}
UE_EVENT_LOCK = threading.Lock()
PRESENCE_RECORD_MODE = False
PRESENCE_VIDEO_DIR = Path("data/presence_videos")

def main():
    rollout_buffer = []
    last_step_cache = None
    video_writer = None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = None
        if POLICY_MODE == "bc":
            model = load_model(str(WEIGHTS_PATH), device=device)
        receive_from_ue(UE_EVENT_LOCK, UE_EVENT_STATE)
        action_cls_model = load_action_cls_model(str(ACTION_CLS_WEIGHTS_PATH), device=device)

        presence_model = load_presence_model(
            str(PRESENCE_WEIGHTS_PATH),
            device=device,
        )
        presence_state = init_presence_state()

        vis_state = None                    
        pol_state = policy_init() 

        frame_ring_buffer = collections.deque(maxlen=CLIP_FRAMES)
        pushed_frames = 0
        recv_frames = 0
        sample_every = 1
        action_lock_until_frame = -1
        locked_action = None
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

                print("[stream] disconnected, closing recorder")
                break
                #continue
            
            recv_frames += 1

            if recv_frames % sample_every != 0:
                continue

            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            
            if PRESENCE_RECORD_MODE:
                if video_writer is None:
                    PRESENCE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

                    timestamp = int(time.time())
                    video_path = (
                        PRESENCE_VIDEO_DIR
                        / f"presence_raw_{timestamp}.mp4"
                    )

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        TARGET_FPS,
                        FRAME_SIZE,
                    )

                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Failed to open video writer: {video_path}"
                        )

                    print(f"[presence-record] started: {video_path}")

                video_writer.write(frame)
            
            frame_ring_buffer.append(frame)
            pushed_frames += 1

            if len(frame_ring_buffer) < CLIP_FRAMES:
                continue
            if pushed_frames % CLIP_STRIDE != 0:
                continue

            frame_id_end = pushed_frames - 1
            frames = build_frame_tensor(frame_ring_buffer)

            presence_prob = infer_player_presence(
                frames=frames,
                model=presence_model,
            )

            presence_raw = int(presence_prob >= 0.60)

            presence_stable = update_presence_state(
                state=presence_state,
                probability=presence_prob,
                enter_threshold=0.60,
                exit_threshold=0.45,
                enter_required=2,
                exit_required=3,
            )

            pred_out = infer_player_state(frames, action_cls_model)
            raw_pred_name = pred_out["pred_name"]
            pred_conf = pred_out["conf"]
            pred_topk_ids = pred_out["topk_ids"][0]
            pred_topk_probs = pred_out["topk_probs"][0]
            pred_topk_names = [pred_out["pred_name"]]  # 先下面再補更完整 mapping
            none_id = CLASS_TO_ID["none"]
            none_prob = pred_out["probs"][0, none_id].item()

            if presence_stable == 1:
                pred_name = raw_pred_name
            else:
                pred_name = "none"

            info, vis_state = vis_update(
                vis_state,
                frames,
                pred_name=pred_name,
                visible=presence_stable,
                frame_id_end=frame_id_end,
            )

            extra_tensor = build_extra_tensor(info, pol_state, frame_id_end)

            with UE_EVENT_LOCK:
                ue_att1_active = UE_EVENT_STATE["att1_active"]
                ue_att1_start = UE_EVENT_STATE["att1_start_pulse"]
                ue_att1_end = UE_EVENT_STATE["att1_end_pulse"]

                ue_att2_active = UE_EVENT_STATE["att2_active"]
                ue_att2_start = UE_EVENT_STATE["att2_start_pulse"]
                ue_att2_end = UE_EVENT_STATE["att2_end_pulse"]

                ue_boss_hit = UE_EVENT_STATE["boss_hit_pulse"]
                ue_player_hit = UE_EVENT_STATE["player_hit_pulse"]
                ue_episode_done = UE_EVENT_STATE["episode_done_flag"]

                # pulse 讀完就清掉
                UE_EVENT_STATE["att1_start_pulse"] = False
                UE_EVENT_STATE["att1_end_pulse"] = False
                UE_EVENT_STATE["att2_start_pulse"] = False
                UE_EVENT_STATE["att2_end_pulse"] = False
                UE_EVENT_STATE["boss_hit_pulse"] = False
                UE_EVENT_STATE["player_hit_pulse"] = False

            if last_step_cache is not None:
                if ue_player_hit:
                    last_step_cache["ue_player_hit_count"] += 1

                if ue_boss_hit:
                    last_step_cache["ue_boss_hit_count"] += 1

                if ue_att1_start:
                    last_step_cache["ue_att1_start"] = True
                if ue_att1_end:
                    last_step_cache["ue_att1_end"] = True

                if ue_att2_start:
                    last_step_cache["ue_att2_start"] = True
                if ue_att2_end:
                    last_step_cache["ue_att2_end"] = True

                if ue_episode_done:
                    last_step_cache["ue_episode_done"] = True
            
            if ue_att1_start:
                print("[UE event] boss normal attack start")
            if ue_att1_end:
                print("[UE event] boss normal attack end")
            if ue_att2_start:
                print("[UE event] boss skill attack1 start")
            if ue_att2_end:
                print("[UE event] boss skill attack1 end")
            if ue_boss_hit:
                print("[UE event] boss hit")
            if ue_player_hit:
                print("[UE event] player hit")
            if ue_episode_done:
                print("[UE event] episode done")
                if last_step_cache is not None:
                    append_cached_step(
                        rollout_buffer,
                        last_step_cache,
                        done=1,
                    )
                    last_step_cache = None

                flush_rollout_buffer(rollout_buffer)

                with UE_EVENT_LOCK:
                    UE_EVENT_STATE["episode_done_flag"] = False

                continue

            if ue_att1_active or ue_att2_active:
                print(f"[decision freeze] attack_active=1 at t={frame_id_end:05d}, skip new inference")
                continue
            if frame_id_end <= action_lock_until_frame:
                print(
                    f"[action freeze] "
                    f"action={locked_action} "
                    f"t={frame_id_end:05d} "
                    f"until={action_lock_until_frame}, "
                    f"skip policy inference and sending"
                )
                continue

            # 已超過鎖定時間，解除鎖定
            if locked_action is not None:
                print(
                    f"[action freeze ended] "
                    f"action={locked_action} "
                    f"at t={frame_id_end:05d}"
                )

                locked_action = None
                action_lock_until_frame = -1

            if last_step_cache is not None:
                append_cached_step(
                    rollout_buffer,
                    last_step_cache,
                    done=0,
                )
                last_step_cache = None

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

                "ue_att1_start": False,
                "ue_att1_end": False,
                "ue_att2_start": False,
                "ue_att2_end": False,
                "ue_boss_hit_count": 0,
                "ue_player_hit_count": 0,
                "ue_episode_done": False,
            }

            if len(rollout_buffer) >= ROLLOUT_SAVE_EVERY or ue_episode_done:
                flush_rollout_buffer(rollout_buffer)

            print(
                f"[t={frame_id_end:05d}] "
                f"presence={presence_prob:.2f} "
                f"raw={presence_raw} "
                f"stable={presence_stable} "
                f"streak=({presence_state['present_streak']},"
                f"{presence_state['absent_streak']}) "
                f"| raw_pred={raw_pred_name}({pred_conf:.2f}) "
                f"effective_pred={pred_name} "
                f"none_prob={none_prob:.2f} "
                f"visible={info['visible']} "
                f"phase={info['phase']} "
                f"hint={info['search_hint']} "
                f"motion={info['motion']:.4f} "
                f"→ bc_action={proposed_action}({action_conf:.2f}) "
                f"final_action={action} "
                f"fire@{fire_frame} "
                f"hold_until={pol_state['hold_until_frame']} "
                f"topk={list(zip(topk_actions, topk_confs))}"
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

            if (
                action in BLOCKING_ACTIONS
                and fire_frame is not None
                and pol_state["hold_until_frame"] is not None
            ):
                locked_action = action
                action_lock_until_frame = int(
                    pol_state["hold_until_frame"]
                )

                print(
                    f"[action freeze started] "
                    f"action={locked_action} "
                    f"fire_frame={fire_frame} "
                    f"until={action_lock_until_frame}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()
            print("[presence-record] video saved")

        cv2.destroyAllWindows()

        if rollout_buffer:
            flush_rollout_buffer(rollout_buffer)
            print("[rollout] flushed remaining buffer on shutdown")

def init_presence_state() -> dict:
    return {
        "visible": 0,
        "present_streak": 0,
        "absent_streak": 0,
    }


def update_presence_state(
    state: dict,
    probability: float,
    enter_threshold: float = 0.60,
    exit_threshold: float = 0.25,
    enter_required: int = 2,
    exit_required: int = 2,
) -> int:
    """
    raw present:
      probability >= 0.60

    raw absent:
      probability < 0.25

    0.25~0.60:
      模糊區域，暫時維持原 visible
    """

    if probability >= enter_threshold:
        state["present_streak"] += 1
        state["absent_streak"] = 0

    elif probability < exit_threshold:
        state["absent_streak"] += 1
        state["present_streak"] = 0

    else:
        # 落在 hysteresis 中間區，不切換狀態
        state["present_streak"] = 0
        state["absent_streak"] = 0

    if state["visible"] == 0:
        if state["present_streak"] >= enter_required:
            state["visible"] = 1
            state["present_streak"] = 0

    else:
        if state["absent_streak"] >= exit_required:
            state["visible"] = 0
            state["absent_streak"] = 0

    return state["visible"]

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