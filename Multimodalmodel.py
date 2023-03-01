import torch
import torch.nn as nn


class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=int((kernel_size-1)/2)),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x

class Multimodalmodel(nn.Module):
    def __init__(self, dim = 512, depth = 5, kernel_size=5, patch_size=7, n_classes=3):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=kernel_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(35, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 512)
        )
        self.head = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x1, x2):
        x1 = self.conv2d1(x1)
        for ConvMixer_block in self.ConvMixer_blocks:
            x1 = ConvMixer_block(x1)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.head(x)
        return x