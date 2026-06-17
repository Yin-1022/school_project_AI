from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def discounted_returns(rewards, dones, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running = 0.0
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns

class RolloutValueDataset(Dataset):
    def __init__(self, rollout_dir, gamma=0.99):
        self.samples = []
        rollout_dir = Path(rollout_dir)
        files = sorted(rollout_dir.glob("*.npz"))
        if not files:
            raise ValueError(f"No rollout files found in {rollout_dir}")

        for path in files:
            with np.load(path, allow_pickle=True) as data:
                frames = data["frames"].astype(np.float32)
                extra = data["extra"].astype(np.float32)
                rewards = data["reward"].astype(np.float32)
                dones = data["done"].astype(np.int64)

                returns = discounted_returns(rewards, dones, gamma=gamma)

                for t in range(len(rewards)):
                    self.samples.append((
                        frames[t],
                        extra[t],
                        returns[t],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, extra, target_return = self.samples[idx]
        return (
            torch.from_numpy(frames),
            torch.from_numpy(extra),
            torch.tensor(target_return, dtype=torch.float32),
        )
    
# ds = RolloutValueDataset("data/rollouts", gamma=0.99)
# print(len(ds))

# frames, extra, target_return = ds[0]
# print(frames.shape)
# print(extra.shape)
# print(target_return.shape)
# print(target_return)