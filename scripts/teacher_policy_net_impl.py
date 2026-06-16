from models import TeacherPolicyNet
from read_teacher_dataset import TeacherDataset
from torch.utils.data import DataLoader
import torch

dataset = TeacherDataset("data/teacher_samples")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

frames, extra, action_id = next(iter(loader))

model = TeacherPolicyNet(in_ch=3, extra_dim=24, num_actions=10)
logits = model(frames, extra)

print("frames:", frames.shape)
print("extra:", extra.shape)
print("logits:", logits.shape)