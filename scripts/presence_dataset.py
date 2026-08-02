from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class PresenceDataset(Dataset):
    def __init__(self, split_dir: str):
        self.root = Path(split_dir)

        self.files = []
        self.labels = []
        self.exposures = []

        exposure_names = ["normal", "medium", "dark"]

        for exposure in exposure_names:
            present_dir = self.root / exposure / "present"
            absent_dir = self.root / exposure / "absent"

            for path in sorted(present_dir.glob("*.npz")):
                self.files.append(path)
                self.labels.append(1)
                self.exposures.append(exposure)

            for path in sorted(absent_dir.glob("*.npz")):
                self.files.append(path)
                self.labels.append(0)
                self.exposures.append(exposure)

        if not self.files:
            raise RuntimeError(
                f"No presence samples found under: {self.root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        label = self.labels[index]
        exposure = self.exposures[index]

        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"].astype(np.float32)

        if frames.shape != (3, 8, 192, 192):
            raise ValueError(
                f"{path} has invalid frames shape: {frames.shape}"
            )

        frames_tensor = torch.from_numpy(frames)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {
            "frames": frames_tensor,
            "label": label_tensor,
            "exposure": exposure,
            "path": str(path),
        }