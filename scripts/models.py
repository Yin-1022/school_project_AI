# scripts/models.py
import torch, torch.nn as nn

class Small3DNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # 時間不降，空間降半

            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),  # 時間降半(8→4)，空間再降

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))      # 聚合到 1×1×1
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):      # x: B,C,T,H,W
        z = self.features(x)   # B,128,1,1,1
        z = z.flatten(1)       # B,128
        return self.head(z)
