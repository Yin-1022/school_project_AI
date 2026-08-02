from collections import deque
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch

from observation_builder import build_frame_tensor
from presence_data import save_presence_sample

CLIP_FRAMES = 8
CLIP_STRIDE = 4
LABEL_STRIDE = 4
FRAME_SIZE = (192, 192)


def build_clip_array(frame_buffer: deque) -> np.ndarray:
    frames = np.stack(list(frame_buffer), axis=0)

    # OpenCV 為 BGR，轉成 RGB
    frames = frames[:, :, :, ::-1].copy()

    # (T, H, W, C) -> (C, T, H, W)
    frames = np.transpose(frames, (3, 0, 1, 2))

    return frames.astype(np.float32) / 255.0


def draw_label_help(
    frame: np.ndarray,
    frame_id_end: int,
) -> np.ndarray:
    preview = frame.copy()

    cv2.putText(
        preview,
        f"frame={frame_id_end}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "P: present | N: absent | S: skip | Q: quit",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return preview


def main(video_path: str) -> None:
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_buffer = deque(maxlen=CLIP_FRAMES)
    frame_index = -1
    clip_count = 0

    while True:
        ok, frame = capture.read()

        if not ok:
            print("[presence-label] reached end of video")
            break

        frame_index += 1

        frame = cv2.resize(
            frame,
            FRAME_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        frame_buffer.append(frame)

        if len(frame_buffer) < CLIP_FRAMES:
            continue

        if (frame_index + 1) % LABEL_STRIDE != 0:
            continue

        clip_preview = list(frame_buffer)

        for preview_frame in clip_preview:
            preview = draw_label_help(preview_frame, frame_index)
            cv2.imshow("Presence Labeling", preview)

            # 每幀顯示約 80 ms，8 幀大約 0.64 秒
            if cv2.waitKey(80) & 0xFF == ord("q"):
                capture.release()
                cv2.destroyAllWindows()
                return

        # 播完後停在最後一幀，等待標註
        preview = draw_label_help(clip_preview[-1], frame_index)
        cv2.imshow("Presence Labeling", preview)
        key = cv2.waitKey(0) & 0xFF

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print("[presence-label] stopped")
            break

        if key == ord("s"):
            continue

        if key not in {ord("p"), ord("n")}:
            continue

        present = key == ord("p")

        frames_tensor = build_frame_tensor(frame_buffer)

        if frames_tensor.ndim == 5:
            if frames_tensor.shape[0] != 1:
                raise ValueError(
                    f"Unexpected frames shape: {frames_tensor.shape}"
                )

            frames_tensor = frames_tensor[0]

        frames = frames_tensor.detach().cpu().numpy()

        save_presence_sample(
            frames=frames,
            present=present,
            frame_id_end=frame_index,
        )

        clip_count += 1

        print(
            f"[presence-label] saved={clip_count} "
            f"label={'present' if present else 'absent'} "
            f"frame={frame_index}"
        )

    capture.release()
    cv2.destroyAllWindows()

    print(f"[presence-label] total saved: {clip_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to recorded Boss-camera video",
    )

    args = parser.parse_args()
    main(args.video_path)