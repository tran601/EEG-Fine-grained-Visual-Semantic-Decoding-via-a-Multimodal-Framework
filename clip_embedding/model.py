import torch
import torch.nn as nn
from einops import rearrange


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(512, 768 * 77)

    def forward(self, x):
        # x shape:[b, 440, 128]
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))  # [b, 1, 440, 128]
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.spatial_temporal(x).view(batch_size, -1)
        y = self.fc(x).view(batch_size, 77, 768)
        return y

class Model_NoSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(512, 768 * 77)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))
        x = self.temporal(x)
        x = self.spatial_temporal(x).view(batch_size, -1)
        y = self.fc(x).view(batch_size, 77, 768)
        return y


class Model_NoTemporal(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(512, 768 * 77)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))
        x = self.spatial(x)
        x = self.spatial_temporal(x).view(batch_size, -1)
        y = self.fc(x).view(batch_size, 77, 768)
        return y



class Model_NoST(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 768 * 77)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.global_pool(x).view(batch_size, -1)  # [b, 256]

        y = self.fc(x).view(batch_size, 77, 768)
        return y


class Model_OnlyTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 768 * 77)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))
        x = self.temporal(x)  
        x = self.global_pool(x).view(batch_size, -1)
        y = self.fc(x).view(batch_size, 77, 768)
        return y


class Model_OnlySpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 768 * 77)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, x.size(1), x.size(2))
        x = self.spatial(x) 
        x = self.global_pool(x).view(batch_size, -1)
        y = self.fc(x).view(batch_size, 77, 768)
        return y
