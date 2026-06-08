import cv2
from pathlib import Path
import collections
import torch
from models import Small3DNet
import numpy as np
from visibility import update as vis_update
from policy import init_state as policy_init, step as policy_step
import socket
import time
from constant import ACTION_ID_TO_NAME, OUT_DIR, PHASE_TO_ONEHOT, SEARCH_HINT_TO_ONEHOT
from pythonosc.udp_client import SimpleUDPClient

_OSC_CLIENT = None

RAW_DIR = Path("data/raw_videos")
video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_action_cls.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
SEQ = 0
_UDP_SOCK = None

def main():
    vis_state = None                    
    pol_state = policy_init() 
    
    model = load_model(str(WEIGHTS_PATH), device="cuda" if torch.cuda.is_available() else "cpu")

    frame_ring_buffer = collections.deque(maxlen=CLIP_FRAMES)
    # cap = cv2.VideoCapture(str(video_path))
    # if not cap.isOpened():
    #     print(f"[warn] cannot open source video: {video_path}")
    #     return []  

    # src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS       #取OpenCV宣告影片的FPS。
    # frame_interval = max(1, round(src_fps / TARGET_FPS))    #幀取樣間隔，例如每五幀取一幀，至少一幀。
    # frame_ring_buffer = collections.deque(maxlen=8)

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

        if len(frame_ring_buffer) == CLIP_FRAMES and pushed_frames % CLIP_STRIDE == 0:
            frames = build_frame_tensor(frame_ring_buffer)
            output = infer_clip(frames, model, (pushed_frames - 1))

            info, vis_state = vis_update(
                vis_state,
                frames,
                output["pred_name"],
                output["visible"],
                output["frame_id_end"]
            )

            action, pol_state, params, fire_frame = policy_step(
                pol_state,
                pred_name=info["pred_name"],
                conf=output["conf"],
                visible=info["visible"],
                phase=info["phase"],
                search_hint=info["search_hint"],
                frame_id_end=output["frame_id_end"]
            )
            extra_tensor = build_extra_tensor(info, pol_state, output["frame_id_end"])
            save_teacher_sample(frames, extra_tensor, action, output["frame_id_end"])

            print(
                f"[t={output['frame_id_end']:05d}] "
                f"pred={output['pred_name']}({output['conf']:.2f}) "
                f"vis={info['visible']} phase={info['phase']} "
                f"hint={info['search_hint']} motion={info['motion']:.4f} "
                f"→ action={action} params={params} fire@{fire_frame} hold_until={pol_state['hold_until_frame']}"
            )

            jsonMsg = {
                "type": "boss_action",
                "ts_frame": output["frame_id_end"],
                "fire_frame": fire_frame,
                "hold_until": pol_state["hold_until_frame"],
                "action": action,
                "params": params,
                "meta": {
                    "pred": output["pred"],
                    "conf": output["conf"],
                    "phase": info["phase"],
                    "search_hint": info["search_hint"]
                },
                "seq": next_seq()
            }

            if fire_frame is not None:
                send_action(jsonMsg)
    # while True:
    #     ok, frame = cap.read()
    #     if not ok:
    #             break
    #     if idx % frame_interval == 0:
    #         frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA) #插值法，即用周圍像素去「估計」新像素值，INTER_AREA適合縮小圖像
    #         frame_ring_buffer.append(frame)
    #         pushed_frames += 1

    #         if len(frame_ring_buffer) == 8 and pushed_frames % CLIP_STRIDE == 0:
    #             frames = list(frame_ring_buffer)
    #             frames = np.stack(frames, axis=0).astype(np.float32) / 255.0  # T,H,W,C
    #             frames = np.transpose(frames, (3,0,1,2))  # C,T,H,W
    #             frames = torch.tensor(frames).unsqueeze(0)  # Add batch dimension: 1,C,T,H,W
    #             output = infer_clip(frames, model, (pushed_frames - 1))
    #             info, vis_state = vis_update(
    #                 vis_state, 
    #                 frames, 
    #                 output["pred_name"], 
    #                 output["visible"], 
    #                 output["frame_id_end"]
    #             )
    #             action, pol_state, params, fire_frame = policy_step(
    #                 pol_state,
    #                 pred_name=info["pred_name"],         
    #                 conf=output["conf"],
    #                 visible=info["visible"],
    #                 phase=info["phase"],
    #                 search_hint=info["search_hint"],
    #                 frame_id_end=output["frame_id_end"]
    #             )
    #             print(f"frame_id_end: {output['frame_id_end']}, pred_name: {output['pred_name']} conf: {output['conf']:.4f}, visible: {output['visible']}")
    #             print(
    #                 f"[t={output['frame_id_end']:05d}] "
    #                 f"pred={output['pred_name']}({output['conf']:.2f}) "
    #                 f"vis={info['visible']} phase={info['phase']} "
    #                 f"hint={info['search_hint']} motion={info['motion']:.4f} "
    #                 f"→ action={action} params={params} fire@{fire_frame} hold_until={pol_state['hold_until_frame']}"
    #             )
    #             jsonMsg = {
    #                 "type": "boss_action",
    #                 "ts_frame": output["frame_id_end"],              # 這次事件的尾幀
    #                 "fire_frame": fire_frame,            # 何時生效（frame_id_end + RT_FRAMES）
    #                 "hold_until": pol_state["hold_until_frame"],            # 最短持有到幀
    #                 "action": action,           # 指令名稱
    #                 "params": params, # Strafe/Search 用
    #                 "meta": {
    #                     "pred": output["pred"],
    #                     "conf": output["conf"],
    #                     "phase": info["phase"],
    #                     "search_hint": info["search_hint"]
    #                 },
    #                 "seq": next_seq()
    #             }
    #             if fire_frame is not None:
    #                 send_action(jsonMsg)
    #     idx += 1
        
    # cap.release()

def infer_clip(frames, model, frame_id_end):
    # Process the 8 frames for inference
    input_tensor = frames.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred = probs.argmax(dim=1).item()
    conf = probs[0, pred].item()
    pred_name = id_to_name(pred)        # 總是給出類別名稱
    visible = 1                         # 先當作可見，交給 3C 的 motion gate 來關掉
    return {"frame_id_end": frame_id_end, "pred": pred, "pred_name": pred_name, "conf": conf, "visible": visible, "outputs": outputs}
        
def id_to_name(pred_id):
    id_to_class = {
        0: "idle",
        1: "move",
        2: "attack",
        3: "roll",
        4: "none",
        5: "jump"
    }
    return id_to_class.get(pred_id, "")

def load_model(weights_path:str, device:str ="cuda"):
    model = Small3DNet(in_ch=3, num_classes=6)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def get_osc_client():
    global _OSC_CLIENT
    if _OSC_CLIENT is None:
        _OSC_CLIENT = SimpleUDPClient("127.0.0.1", 9999)
    return _OSC_CLIENT

def send_action(msg):
    client = get_osc_client()

    args = [
        msg["action"],                          # string
        int(msg["ts_frame"]),                   # int
        int(msg["fire_frame"]),                 # int
        int(msg["hold_until"]),                 # int
        float(msg["meta"]["conf"]),             # float
        str(msg["meta"]["phase"]),              # string
        str(msg["meta"]["search_hint"] or ""),  # string
        int(msg["seq"]),                        # int
    ]

    client.send_message("/boss/action", args)
    
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

                # if debug_show:
                #     cv2.imshow("UE Boss Vision", frame_bgr)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         raise KeyboardInterrupt

                conn.sendall(b"OK")
                yield frame_bgr

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            print("[系統] 目前連線已中斷，回到監聽狀態")
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

def save_teacher_sample(frames, extra, action_name, frame_id_end):
    out_dir = Path("data/teacher_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = frames.squeeze(0).detach().cpu().numpy()   # shape (C,T,H,W)
    extra = extra.squeeze(0).detach().cpu().numpy()     # shape (24,)
    if action_name not in ACTION_NAME_TO_ID:
        raise ValueError(f"Unknown action_name: {action_name}")
    action_id = ACTION_NAME_TO_ID[action_name]

    timestamp = int(time.time() * 1000)
    out_path = out_dir / f"sample_{timestamp}_{frame_id_end:06d}.npz"

    np.savez(
        out_path, 
        frames=frames, 
        extra=extra, 
        action_id=np.int64(action_id),
        action_name=action_name,
        frame_id_end=np.int64(frame_id_end),
    )
    print(f"Saved teacher sample to {out_path}")
    
    
if __name__ == "__main__":
    main()