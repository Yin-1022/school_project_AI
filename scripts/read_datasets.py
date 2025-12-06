# scripts/datasets.py
import os, cv2, numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path

CLASS_TO_ID = {"idle":0, "move":1, "attack":2, "roll":3, "none":4, "jump":5}

def _read_clip_frames(clip_path: str, expected_frames=8, size=(160,160)):

    path = Path(clip_path)
    frames = []

    if path.is_dir():
        imgs = sorted(path.glob("frame_*.png"))
        for img in imgs[:expected_frames]:
            im = cv2.imread(str(img))
            if im is None:  # 壞檔就跳過
                continue
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)  # size=(W,H)
            frames.append(im)
    else:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            print(f"[info] reading video: {path}")
        count = 0
        while count < expected_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            count += 1
        cap.release()

    # --------- 這裡是重點：不足幀就補齊 ----------
    if len(frames) == 0:
        # 全黑占位：注意 numpy 需要 (H,W,3)
        W, H = size
        frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(expected_frames)]
    elif len(frames) < expected_frames:
        last = frames[-1]
        frames.extend([last.copy() for _ in range(expected_frames - len(frames))])
    # ---------------------------------------------

    arr = np.stack(frames[:expected_frames], axis=0).astype(np.float32) / 255.0  # T,H,W,C
    arr = np.transpose(arr, (3,0,1,2))  # C,T,H,W
    return arr

# 放在檔案最上方的 import 區已經有就不用重複
from pathlib import Path
import os, cv2

class ClipDataset(Dataset):
    def __init__(self, csv_path, root="", expected_frames=8, size=(160,160), augment=False):
        self.root = Path(root) if root else None
        self.expected_frames = expected_frames
        self.size = size
        self.augment = augment

        # 讀 CSV -> 暫存原始 items
        self.items = []  # (clip_path_str, label_id)
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            next(f)  # 跳過表頭: clip_path,label
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_path, label = line.split(",")
                if label not in CLASS_TO_ID:
                    raise ValueError(f"Unknown label '{label}' in CSV; expected one of {list(CLASS_TO_ID.keys())}")
                self.items.append((raw_path, CLASS_TO_ID[label]))

        # ===== 這裡開始是「步驟 1：路徑清洗 + 自動修正 + 篩壞檔」 =====
        def _clean(s: str):
            # 去 BOM/引號/多餘空白
            return s.strip().strip('"').strip("'")

        def _resolve_any(p: Path):
            # 0) 直接存在就回傳
            if p.exists():
                return p

            # 1) mp4 -> avi
            if p.suffix.lower() == ".mp4":
                q = p.with_suffix(".avi")
                if q.exists():
                    return q

            # 2) 影格資料夾（去掉副檔名）
            r = p.with_suffix("")
            if r.exists() and r.is_dir():
                return r

            # 3) 推測 clips 子路徑
            #    取出檔名與 stem（raw_video_1_clip_00000 -> stem_base=raw_video_1）
            name = p.name                           # raw_video_1_clip_00000.mp4
            stem_full = p.stem                      # raw_video_1_clip_00000
            stem_base = stem_full.split("_clip_")[0]  # raw_video_1
            base_dir = p.parent                     # 目前是 <root>/data 或 <root>/data/clips 或空

            # 3a) base/clips/<stem_base>/<name>（原副檔名）
            guess = base_dir / "clips" / stem_base / name
            if guess.exists():
                return guess

            # 3b) 同路徑但 .avi
            if guess.suffix.lower() == ".mp4":
                guess_avi = guess.with_suffix(".avi")
                if guess_avi.exists():
                    return guess_avi

            # 3c) 影格資料夾：base/clips/<stem_base>/<stem_full>/
            guess_dir = base_dir / "clips" / stem_base / stem_full
            if guess_dir.exists() and guess_dir.is_dir():
                return guess_dir

            # 3d) 若 CSV 已經含 clips/但少了中間資料夾（clips/raw_video_1_clip_00000.mp4）
            if "clips" in str(p):
                parent = p.parent
                guess2 = parent / stem_base / p.name
                if guess2.exists():
                    return guess2
                if p.suffix.lower() == ".mp4":
                    guess2_avi = guess2.with_suffix(".avi")
                    if guess2_avi.exists():
                        return guess2_avi
                guess2_dir = parent / stem_base / p.stem
                if guess2_dir.exists() and guess2_dir.is_dir():
                    return guess2_dir

            return None



        BASE_DIR = Path(__file__).resolve().parents[1]  # 專案根 (…/your_project)
        base = Path(root) if root else BASE_DIR

        fixed_items = []
        total_csv = len(self.items)
        exist_ok = 0
        open_ok = 0
        dropped = 0

        for raw_path, y in self.items:
            p = base / _clean(raw_path)
            p = p.resolve()

            p2 = _resolve_any(p)
            if p2 is None:
                dropped += 1
                continue
            exist_ok += 1

            ok = False
            if p2.is_dir():
                # 影格資料夾：至少要有 expected_frames 張
                ok = len(list(p2.glob("frame_*.png"))) >= self.expected_frames
            else:
                # 影片檔案：試著打開並讀到 expected_frames 幀
                cap = cv2.VideoCapture(str(p2), cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(str(p2), cv2.CAP_ANY)
                if cap.isOpened():
                    cnt = 0
                    while cnt < self.expected_frames:
                        ret, _ = cap.read()
                        if not ret:
                            break
                        cnt += 1
                    cap.release()
                    ok = (cnt >= self.expected_frames)

            if ok:
                open_ok += 1
                fixed_items.append((str(p2), y))
            else:
                dropped += 1

        self.items = fixed_items
        print(f"[dataset] total_csv={total_csv} exist={exist_ok} openable={open_ok} dropped={dropped}")
        # ===== 「步驟 1」到這裡結束 =====

        if dropped and exist_ok == 0:
            print("[hint] Example of unresolved path (first 5 from CSV):")
            # 注意：此時 self.items 已經是 fixed_items（可讀清單）；我們想看原 CSV 的前幾條
            # 所以改印「原始 CSV 的前幾條」來判斷 path 基底是否對
            sample_csv = []
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                next(f)
                for _ in range(5):
                    line = f.readline()
                    if not line: break
                    sample_csv.append(line.strip().split(",")[0])
            base = Path(root) if root else Path(__file__).resolve().parents[1]
            for raw in sample_csv:
                p_try = (base / raw).resolve()
                print("  - tried:", p_try)


    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        x = _read_clip_frames(path, self.expected_frames, self.size)  # C,T,H,W

        # 簡單增強（只在訓練用）：左右翻轉 + 亮度抖動（輕微）
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:,:, :, ::-1].copy()
            if np.random.rand() < 0.3:
                factor = 1.0 + np.random.uniform(-0.1, 0.1)
                x = np.clip(x * factor, 0.0, 1.0)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
