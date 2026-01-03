import cv2
from pathlib import Path
import collections
import torch
from models import Small3DNet
import numpy as np
from visibility import update as vis_update, stateInit as vis_init
from policy import init_state as policy_init, step as policy_step

RAW_DIR = Path("data/raw_videos")
video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_action_cls.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長（重疊有助抓動作起迄）
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)

def main():
    vis_state = None                    
    pol_state = policy_init() 
    
    model = load_model(str(WEIGHTS_PATH), device="cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] cannot open source video: {video_path}")
        return []  

    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS       #取OpenCV宣告影片的FPS。
    frame_interval = max(1, round(src_fps / TARGET_FPS))    #幀取樣間隔，例如每五幀取一幀，至少一幀。
    frame_ring_buffer = collections.deque(maxlen=8)

    idx = 0
    pushed_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
                break
        if idx % frame_interval == 0:
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA) #插值法，即用周圍像素去「估計」新像素值，INTER_AREA適合縮小圖像
            frame_ring_buffer.append(frame)
            pushed_frames += 1

            if len(frame_ring_buffer) == 8 and pushed_frames % CLIP_STRIDE == 0:
                frames = list(frame_ring_buffer)
                frames = np.stack(frames, axis=0).astype(np.float32) / 255.0  # T,H,W,C
                frames = np.transpose(frames, (3,0,1,2))  # C,T,H,W
                frames = torch.tensor(frames).unsqueeze(0)  # Add batch dimension: 1,C,T,H,W
                output = infer_clip(frames, model, (pushed_frames - 1))
                info, vis_state = vis_update(
                    vis_state, 
                    frames, 
                    output["pred_name"], 
                    output["visible"], 
                    output["frame_id_end"]
                )
                cmd, pol_state, params, fire_frame = policy_step(
                    pol_state,
                    pred_name=info["pred_name"],          # 可能已被 motion gate 覆寫
                    conf=output["conf"],
                    visible=info["visible"],
                    phase=info["phase"],
                    search_hint=info["search_hint"],
                    frame_id_end=output["frame_id_end"]
                )
                print(f"frame_id_end: {output['frame_id_end']}, pred_name: {output['pred_name']} conf: {output['conf']:.4f}, visible: {output['visible']}")
                print(
                    f"[t={output['frame_id_end']:05d}] "
                    f"pred={output['pred_name']}({output['conf']:.2f}) "
                    f"vis={info['visible']} phase={info['phase']} "
                    f"hint={info['search_hint']} motion={info['motion']:.4f} "
                    f"→ cmd={cmd} params={params} fire@{fire_frame} hold_until={pol_state['hold_until_frame']}"
                )       
        idx += 1
        
    cap.release()

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

if __name__ == "__main__":
    main()