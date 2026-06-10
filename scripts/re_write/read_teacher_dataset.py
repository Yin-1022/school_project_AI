from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class TeacherDataset(Dataset):
    def __init__(self, sample_dir):
        self.sample_dir = Path(sample_dir)
        self.files = sorted(self.sample_dir.glob("*.npz"))

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {self.sample_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]

        with np.load(npz_path, allow_pickle=True) as data:
            frames = data["frames"].astype(np.float32)      # (3,8,192,192)
            extra = data["extra"].astype(np.float32)        # (24,)
            action_id = int(data["action_id"])              # scalar int

        frames = torch.from_numpy(frames)                   # float tensor
        extra = torch.from_numpy(extra)                     # float tensor
        action_id = torch.tensor(action_id, dtype=torch.long)

        return frames, extra, action_id
    
# ds = TeacherDataset("data/teacher_samples")
# print(len(ds))

# frames, extra, action_id = ds[0]
# print(frames.shape)
# print(extra.shape)
# print(action_id)
# print(action_id.dtype)