import cv2
from pathlib import Path
import collections
from pythonosc import udp_client, dispatcher, osc_server
import torch
import numpy as np
from models import TeacherPolicyNet
from visibility import update as vis_update
from policy import init_state as policy_init, is_ready, arm_cooldown
import socket
import time
import threading
from constant import (
    ACTION_ID_TO_NAME,
    PHASE_TO_ONEHOT,
    SEARCH_HINT_TO_ONEHOT,
    RT_FRAMES,
    MIN_HOLD_FRAMES,
    SAME_ACTION_REFIRE_FRAMES,
    CD_EVADE,
    CD_TURN,
    CD_PATROL,
    CD_STRAFE,
)
from pythonosc.udp_client import SimpleUDPClient

_OSC_CLIENT = None

RAW_DIR = Path("data/raw_videos")
video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
SEQ = 0
_UDP_SOCK = None
UE_EVENT_STATE = {
    "attack_triggered": False,
}
UE_EVENT_LOCK = threading.Lock()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(WEIGHTS_PATH), device=device)
    receive_from_ue()

    vis_state = None                    
    pol_state = policy_init() 

    frame_ring_buffer = collections.deque(maxlen=CLIP_FRAMES)
    idx = 0
    pushed_frames = 0
    recv_frames = 0
    sample_every = 1
    global SEQ

    for frame in tcp_frame_stream(host='127.0.0.1', port=9999, img_w=192, img_h=192, img_c=3, debug_show=False):
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

        bc_out = infer_action(frames, extra_tensor, model)
        proposed_action = bc_out["action_name"]
        action_conf = bc_out["conf"]
        topk_actions = [ACTION_ID_TO_NAME[id] for id in bc_out["topk_ids"][0]]
        topk_confs = bc_out["topk_probs"][0]

        with UE_EVENT_LOCK:
            ue_attack_triggered = UE_EVENT_STATE["attack_triggered"]
            if ue_attack_triggered:
                UE_EVENT_STATE["attack_triggered"] = False

        if ue_attack_triggered:
            proposed_action = "Attack"

        action, pol_state, fire_frame = apply_action_with_state(
            pol_state,
            proposed_action=proposed_action,
            topk_actions=topk_actions,
            frame_id_end=frame_id_end,
            info=info
        )

        print(
            f"[t={frame_id_end:05d}] "
            f"vis={info['visible']} phase={info['phase']} "
            f"hint={info['search_hint']} motion={info['motion']:.4f} "
            f"→ bc_action={proposed_action}({action_conf:.2f}) "
            f"final_action={action} fire@{fire_frame} hold_until={pol_state['hold_until_frame']}"
            f" topk={list(zip(topk_actions, topk_confs))}"
        )

        if fire_frame is None:
            continue

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
            "seq": next_seq()
        }

        send_action(jsonMsg)

def load_model(weights_path:str, device:str ="cuda"):
    model = TeacherPolicyNet(in_ch=3, extra_dim=24, num_actions=10)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def infer_action(frames, extra, model):
    device = next(model.parameters()).device
    frames = frames.to(device)
    extra = extra.to(device)
    
    with torch.no_grad():
        logits = model(frames, extra)
        probs = torch.softmax(logits, dim=1)

    action_id = probs.argmax(dim=1).item()
    conf = probs[0, action_id].item()
    action_name = ACTION_ID_TO_NAME[action_id]
    topk_probs, topk_ids = torch.topk(probs, k=3, dim=1)

    return {
        "action_id": action_id,
        "action_name": action_name,
        "conf": conf,
        "logits": logits,
        "probs": probs,
        "topk_ids": topk_ids.cpu().numpy(),
        "topk_probs": topk_probs.cpu().numpy(),
    }

def apply_action_with_state(pol_state, proposed_action, topk_actions, frame_id_end, info):
    hold_until_frame = pol_state.get("hold_until_frame", -1)
    last_action = pol_state.get("last_action", "Hold")
    last_proposed_action = pol_state.get("last_proposed_action", None)
    same_action_streak = pol_state.get("same_action_streak", 0)
    hold_streak = pol_state.get("hold_streak", 0)
    last_non_hold_action = pol_state.get("last_non_hold_action", "Hold")

    if frame_id_end < hold_until_frame:
        if last_action != "Hold":
            return last_action, pol_state, None
    
    if proposed_action == last_proposed_action:
        same_action_streak += 1
    else:
        same_action_streak = 1

    pol_state["last_proposed_action"] = proposed_action
    pol_state["same_action_streak"] = same_action_streak

    action = proposed_action

    if action in {"StrafeRight", "StrafeLeft"} and same_action_streak > 3:
        fallback = None
        for cand in topk_actions:
            if cand != action:
                fallback = cand
                break
        if fallback is not None:
            action = fallback
            pol_state["same_action_streak"] = 1
            pol_state["last_proposed_action"] = action

    if action == "EvadeBack" and not is_ready(pol_state, "EvadeBack", frame_id_end):
        action = "Retreat"
    
    if action in {"SearchTurnLeft", "SearchTurnRight"} and not is_ready(pol_state, "SearchTurn", frame_id_end):
        action = last_action if frame_id_end < hold_until_frame else "Hold"

    if action in {"PatrolStepLeft", "PatrolStepRight"} and not is_ready(pol_state, "PatrolStep", frame_id_end):
        action = last_action if frame_id_end < hold_until_frame else "Hold"

    if action == "Hold":
        if last_action == "Hold":
            hold_streak += 1
        else:
            hold_streak = 1
    else:
        hold_streak = 0
        last_non_hold_action = action

    if action == "Hold" and hold_streak > 2:
        fallback = None
        for cand in topk_actions:
            if cand != "Hold" and cand != last_non_hold_action:
                fallback = cand
                break
        
        if fallback is None:
            for cand in topk_actions:
                if cand != "Hold":
                    fallback = cand
                    break

        if fallback is not None:
            action = fallback
            hold_streak = 0
            last_non_hold_action = action
    
    if action == "Hold":
        pol_state["hold_streak"] = hold_streak
        pol_state["last_non_hold_action"] = last_non_hold_action

        if hold_streak <= 2:
            visible = info.get("visible", 0)
            phase = info.get("phase", "patrol")

            # 1) track：可以沿用 chase 類動作
            if visible == 1 and phase == "track":
                continued_action = last_action
                if continued_action in {
                    "Advance",
                    "StrafeLeft",
                    "StrafeRight",
                }:
                    last_fire_at = pol_state.get("last_action_at_frame", -1)
                    if (frame_id_end - last_fire_at) >= SAME_ACTION_REFIRE_FRAMES:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action_at_frame"] = frame_id_end
                        return continued_action, pol_state, fire_frame
                    return continued_action, pol_state, None

            # 2) reacq：優先 SearchTurn，其次 PatrolStep
            if phase == "reacq":
                for cand in topk_actions:
                    if cand in {"SearchTurnLeft", "SearchTurnRight"}:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action"] = cand
                        pol_state["last_action_at_frame"] = frame_id_end
                        pol_state["hold_until_frame"] = frame_id_end
                        return cand, pol_state, fire_frame

                for cand in topk_actions:
                    if cand in {"PatrolStepLeft", "PatrolStepRight"}:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action"] = cand
                        pol_state["last_action_at_frame"] = frame_id_end
                        pol_state["hold_until_frame"] = frame_id_end
                        return cand, pol_state, fire_frame

                return "Hold", pol_state, None

            # 3) patrol：優先 PatrolStep，不要沿用 Advance
            if phase == "patrol":
                for cand in topk_actions:
                    if cand in {"PatrolStepLeft", "PatrolStepRight"}:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action"] = cand
                        pol_state["last_action_at_frame"] = frame_id_end
                        pol_state["hold_until_frame"] = frame_id_end
                        return cand, pol_state, fire_frame

                return "Hold", pol_state, None

    if action == last_action:
        last_fire_at = pol_state.get("last_action_at_frame", -1)
        if (frame_id_end - last_fire_at) < SAME_ACTION_REFIRE_FRAMES:
            pol_state["hold_streak"] = hold_streak
            pol_state["last_non_hold_action"] = last_non_hold_action
            return action, pol_state, None

    fire_frame = frame_id_end + RT_FRAMES
    if action != "Hold":
        pol_state["last_action"] = action
        pol_state["last_action_at_frame"] = frame_id_end
        pol_state["hold_until_frame"] = fire_frame + MIN_HOLD_FRAMES
    else:
        pol_state["hold_until_frame"] = frame_id_end
    pol_state["hold_streak"] = hold_streak
    pol_state["last_non_hold_action"] = last_non_hold_action

    if action == "EvadeBack":
        arm_cooldown(pol_state, "EvadeBack", fire_frame, CD_EVADE)
    elif action == "SearchTurnLeft" or action == "SearchTurnRight":
        arm_cooldown(pol_state, "SearchTurn", fire_frame, CD_TURN)
    elif action == "PatrolStepLeft" or action == "PatrolStepRight":
        arm_cooldown(pol_state, "PatrolStep", fire_frame, CD_PATROL)

    return action, pol_state, fire_frame

def get_osc_client():
    global _OSC_CLIENT
    if _OSC_CLIENT is None:
        _OSC_CLIENT = SimpleUDPClient("127.0.0.1", 12345)
    return _OSC_CLIENT

def send_action(msg):
    client = get_osc_client()

    action_name = msg["action"]
    angle = 0.0

    if action_name == "SearchTurnRight":
        action_name = "SearchTurn"
        angle = 50.0
    elif action_name == "SearchTurnLeft":
        action_name = "SearchTurn"
        angle = -50.0
    elif action_name == "PatrolStepRight":
        action_name = "PatrolStep"
        angle = 50.0
    elif action_name == "PatrolStepLeft":
        action_name = "PatrolStep"
        angle = -50.0

    args = [
        action_name,                            # string
        float(angle),                           # float
        int(msg["ts_frame"]),                   # int
        int(msg["fire_frame"]),                 # int
        int(msg["hold_until"]),                 # int
        float(msg["meta"]["conf"]),             # float
        str(msg["meta"]["phase"]),              # string
        str(msg["meta"]["search_hint"] or ""),  # string
        int(msg["seq"]),                        # int
    ]

    client.send_message("/boss/action", args)

def receive_from_ue():
    def on_attack_start(address, *args):
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["attack_triggered"] = True
        print(f"[← UE] 開始攻擊！args: {args}")

    def on_attack_end(address, *args):
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["attack_triggered"] = False
        print(f"[← UE] 結束攻擊！args: {args}")

    def on_fallback(address, *args):
        print(f"[← UE] 未知訊息 {address}，args: {args}")

    dp = dispatcher.Dispatcher()
    dp.map("/attatart", on_attack_start)
    dp.map("/attend",   on_attack_end)
    dp.set_default_handler(on_fallback)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 12346), dp)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[接收] 監聽 port {12346}...")

def next_seq():
    global SEQ
    SEQ += 1
    return SEQ

def tcp_frame_stream(host='127.0.0.1', port=9999, img_w=192, img_h=192, img_c=3, debug_show=False):
    frame_size = img_w * img_h * img_c

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print("=== Python 推論伺服器已就緒 ===")

    while True:
        print(f"[等待中] 正在監聽 Port {port}...")
        conn = None
        try:
            conn, addr = server_socket.accept()
            print(f"[已連線] 與 UE 建立連線: {addr}")

            data_buffer = b""
            while True:
                while len(data_buffer) < frame_size:
                    packet = conn.recv(frame_size - len(data_buffer))
                    if not packet:
                        print("[通知] UE 連線中斷")
                        raise ConnectionResetError
                    data_buffer += packet

                frame_data = data_buffer[:frame_size]
                data_buffer = data_buffer[frame_size:]

                frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((img_h, img_w, img_c))

                # 模型若沿用 OpenCV 訓練資料，建議轉回 BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                conn.sendall(b"OK")
                yield frame_bgr

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            print("[系統] 目前連線已中斷，回到監聽狀態")
        except KeyboardInterrupt:
            print("[系統] 收到中斷事件，忽略這次中斷並回到監聽狀態")
        finally:
            if conn is not None:
                conn.close()
            if debug_show:
                cv2.destroyAllWindows()

ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}

def one_hot_action(action_name: str) -> list[float]:
    vec = [0.0] * len(ACTION_ID_TO_NAME)
    action_id = ACTION_NAME_TO_ID.get(action_name, 0)   # 預設 Hold
    vec[action_id] = 1.0
    return vec

def build_frame_tensor(frame_ring_buffer):
    frames = list(frame_ring_buffer)
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0   # T,H,W,C
    frames = np.transpose(frames, (3,0,1,2))                       # C,T,H,W
    frames_tensor = torch.from_numpy(frames).unsqueeze(0)          # 1,C,T,H,W
    return frames_tensor

def build_extra_tensor(info, pol_state, frame_id_end):
    visible = float(info.get("visible", 0))
    motion = float(info.get("motion", 0.0))

    phase = info.get("phase", "track")
    phase_vec = PHASE_TO_ONEHOT.get(phase, PHASE_TO_ONEHOT["patrol"])

    search_hint = info.get("search_hint", None)
    search_hint_vec = SEARCH_HINT_TO_ONEHOT.get(search_hint, SEARCH_HINT_TO_ONEHOT["none"])

    last_action_name = pol_state.get("last_action", "Hold")
    last_action_vec = one_hot_action(last_action_name)

    last_action_at_frame = pol_state.get("last_action_at_frame", -1)
    if last_action_at_frame < 0:
        time_since_last_action = 0.0
    else:
        time_since_last_action = min(float(frame_id_end - last_action_at_frame) / 10.0, 1.0)

    hold_until_frame = pol_state.get("hold_until_frame", -1)
    hold_active = 1.0 if hold_until_frame > frame_id_end else 0.0
    
    cooldowns = pol_state.get("cooldowns", {})
    evade_ready = 1.0 if frame_id_end >= cooldowns.get("EvadeBack", 0) else 0.0
    turn_ready = 1.0 if frame_id_end >= cooldowns.get("SearchTurn", 0) else 0.0
    patrol_ready = 1.0 if frame_id_end >= cooldowns.get("PatrolStep", 0) else 0.0

    extra = [
        visible,
        motion,
        *phase_vec,
        *search_hint_vec,
        *last_action_vec,
        time_since_last_action,
        hold_active,
        evade_ready,
        turn_ready,
        patrol_ready
    ]
    extra = np.asarray(extra, dtype=np.float32)

    assert extra.shape == (24,), f"extra feature shape mismatch: got {extra.shape}, expected (24,)"
    extra_tensor = torch.from_numpy(extra).unsqueeze(0)   # shape = (1, 24)
    return extra_tensor

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