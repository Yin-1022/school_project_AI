import torch
from torch import nn
import torch.nn.functional as F

class PresenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(128, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.features(frames)

        avg_feature = F.adaptive_avg_pool3d(
            x,
            output_size=(1, 1, 1),
        )

        max_feature = F.adaptive_max_pool3d(
            x,
            output_size=(1, 1, 1),
        )

        x = torch.cat(
            [avg_feature, max_feature],
            dim=1,
        )

        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits.squeeze(1)