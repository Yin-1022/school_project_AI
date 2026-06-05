# scripts/extract_clips.py
import cv2
import constant as const
from pathlib import Path
from tqdm    import tqdm

RAW_DIR = const.RAW_DIR
OUT_DIR = const.OUT_DIR
CLIP_FRAMES     = const.CLIP_FRAMES    # 每個 clip 的影格數
CLIP_STRIDE     = const.CLIP_STRIDE    # 滑窗步長（重疊有助抓動作起迄）
TARGET_FPS      = const.TARGET_FPS     # 轉成固定 FPS，降低抖動與檔案大小
FRAME_SIZE      = const.FRAME_SIZE     # 先取小一點，後面分類器更輕量

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_from_video(video_path: Path):
    name = video_path.stem      # 不含副檔名的檔名
    out_dir = OUT_DIR / name
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] cannot open source video: {video_path}")
        return 0  

    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS       #取OpenCV宣告影片的FPS。
    frame_interval = max(1, round(src_fps / TARGET_FPS))    #幀取樣間隔，例如每五幀取一幀，至少一幀。

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA) #插值法，即用周圍像素去「估計」新像素值，INTER_AREA適合縮小圖像
            frames.append(frame)
        idx += 1
    cap.release()

    # 若不足一個 clip，直接回 0
    if len(frames) < CLIP_FRAMES:
        return 0

    # 建議改成 MJPEG .avi 比較穩
    fourcc = cv2.VideoWriter_fourcc(*"MJPG") #四個字元指定影片編碼，選擇Motion JPEG
    count = 0
    for start_index in range(0, len(frames) - CLIP_FRAMES + 1, CLIP_STRIDE): #例如0~50-8+1，每次跳4，產生重疊或不重疊的 clip
        clip = frames[start_index:start_index+CLIP_FRAMES]
        out_path = out_dir / f"{name}_clip_{count:05d}.avi"
        vw = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, FRAME_SIZE)
        for f in clip:
            vw.write(f)
        vw.release()
        count += 1

    return int(count)  # << 明確回整數

def main():
    ensure_dir(OUT_DIR)
    videos = sorted([p for p in RAW_DIR.glob("*.mp4")]) #找所有副檔名為 .mp4 的檔案
    if not videos:
        print("No videos in data/raw_videos. Put some .mp4 there.")
        return
    total = 0
    for v in tqdm(videos, desc="Extracting"):
        n = extract_from_video(v)
        if n is None:
            n = 0
        total += n
    print(f"Extracting completes. Total clips: {total}")

if __name__ == "__main__":
    main()
