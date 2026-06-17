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
    
class TeacherPolicyNet(nn.Module):
    def __init__(self, in_ch=3, extra_dim=24, num_actions=10):
        super().__init__()

        # Sequential 將多個層（Layer）或模組（Module）按順序組合在一起
        self.visual = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, stride=1, padding=1), # 3D 卷積層看C,T,H,W 
            nn.BatchNorm3d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),  

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))      # 聚合到 1×1×1
        )

        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )

    def forward(self, frames, extra):
        v = self.visual(frames)   # B,128,1,1,1
        v = v.flatten(1)         # B,128

        e = self.extra_mlp(extra) # B,64

        z = torch.cat([v, e], dim=1) # B,192
        return self.head(z)      # B,num_actions

class TeacherValueNet(nn.Module):
    def __init__(self, in_ch=3, extra_dim=24):
        super().__init__()

        self.visual = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, frames, extra):
        v = self.visual(frames)
        v = v.flatten(1)

        e = self.extra_mlp(extra)

        z = torch.cat([v, e], dim=1)
        value = self.head(z).squeeze(1)   # [B]
        return value

class TeacherActorCriticNet(nn.Module):
    def __init__(self, in_ch=3, extra_dim=24, num_actions=10):
        super().__init__()

        self.visual = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, frames, extra):
        v = self.visual(frames).flatten(1)
        e = self.extra_mlp(extra)
        z = torch.cat([v, e], dim=1)
        h = self.trunk(z)

        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(1)
        return logits, value
        
class ExtraOnlyPolicyNet(nn.Module):
    def __init__(self, extra_dim=24, num_actions=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),
        )

    def forward(self, frames, extra):
        return self.net(extra)