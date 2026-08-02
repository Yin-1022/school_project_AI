from collections import deque
from pathlib import Path
import argparse

import cv2
import numpy as np


def tensor_frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    """
    frame shape: (3, H, W)
    預期數值範圍為 [0, 1]
    """

    frame = np.transpose(frame, (1, 2, 0))
    frame = np.clip(frame, 0.0, 1.0)
    frame = (frame * 255.0).astype(np.uint8)

    # RGB -> BGR
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def play_clip(
    frames: np.ndarray,
    filename: str,
    delay_ms: int = 120,
) -> int:
    """
    frames shape: (3, 8, 192, 192)

    回傳：
    q：結束
    n：下一個
    r：重播
    """

    while True:
        for time_index in range(frames.shape[1]):
            frame = frames[:, time_index]
            preview = tensor_frame_to_bgr(frame)

            cv2.putText(
                preview,
                filename,
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"frame {time_index + 1}/{frames.shape[1]}",
                (5, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            enlarged = cv2.resize(
                preview,
                (576, 576),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imshow("Presence Error Viewer", enlarged)

            key = cv2.waitKey(delay_ms) & 0xFF

            if key == ord("q"):
                return ord("q")

        # 播放完後停在最後一幀
        cv2.putText(
            enlarged,
            "N: next | R: replay | Q: quit",
            (10, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Presence Error Viewer", enlarged)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            return ord("q")

        if key == ord("n"):
            return ord("n")

        if key == ord("r"):
            continue


def main(error_dir: str) -> None:
    root = Path(error_dir)
    files = sorted(root.rglob("*.npz"))

    if not files:
        raise RuntimeError(f"No npz files found under: {root}")

    print(f"found {len(files)} error samples")

    for index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"].astype(np.float32)

        if frames.shape != (3, 8, 192, 192):
            print(f"[skip] invalid shape {frames.shape}: {path}")
            continue

        print(f"[{index}/{len(files)}] {path}")

        key = play_clip(
            frames=frames,
            filename=path.name,
        )

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "error_dir",
        type=str,
        help="Error sample directory",
    )

    args = parser.parse_args()
    main(args.error_dir)