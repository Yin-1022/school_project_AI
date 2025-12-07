import cv2
from pathlib import Path
import collections
import torch
from models import Small3DNet
import numpy as np

RAW_DIR = Path("data/raw_videos")
video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_action_cls.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長（重疊有助抓動作起迄）
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
ms_per_frame = 1000 / TARGET_FPS
delta_t_ms = (CLIP_STRIDE / TARGET_FPS) * 1000

def main():
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
                print(f"frame_id_end: {output['frame_id_end']}, pred_name: {output['pred_name']} conf: {output['conf']:.4f}, visible: {output['visible']}")       
        idx += 1
        
    cap.release()

def infer_clip(frames, model, frame_id_end):
    # Process the 8 frames for inference
    input_tensor = frames.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred = probs.argmax(dim=1).item()
    conf = probs.max().item()
    pred_name = id_to_name(pred) if conf > 0.5 else "none"
    visible = 1 if pred_name != "none" else 0
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