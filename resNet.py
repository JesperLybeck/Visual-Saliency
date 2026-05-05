import torch
import torch.nn as nn
import torch.nn.functional as F

class MainPath(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super().__init__()
        F1, F2, F3 = filters
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, F1, kernel_size=1, stride=stride),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(F1, F2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F3, kernel_size=1),
            nn.BatchNorm2d(F3),
        )
        # weight init (simple)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.main_path(x)

class IdentityBlock(MainPath):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main_path(x) + x)

class ConvolutionalBlock(MainPath):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size, stride=2)
        self.relu = nn.ReLU()
        F3 = filters[2]
        self.shortcut_path = nn.Sequential(
            nn.Conv2d(in_channels, F3, kernel_size=1, stride=2),
            nn.BatchNorm2d(F3),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.main_path(x) + self.shortcut_path(x)
        return self.relu(y)

class SaliencyResNet(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ConvolutionalBlock(64, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
        )
        self.stage3 = nn.Sequential(
            ConvolutionalBlock(256, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
        )
        self.stage4 = nn.Sequential(
            ConvolutionalBlock(512, [256, 256, 1024], kernel_size=3),
            *[IdentityBlock(1024, [256, 256, 1024], kernel_size=3) for _ in range(5)]
        )
        self.stage5 = nn.Sequential(
            ConvolutionalBlock(1024, [512, 512, 2048], kernel_size=3),
            IdentityBlock(2048, [512, 512, 2048], kernel_size=3),
            IdentityBlock(2048, [512, 512, 2048], kernel_size=3),
        )

        # Small conv head -> 1 channel, then upsample to input size
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        x = self.stem(x)     # downsample
        x = self.stage2(x)   # /4 -> /8 depending on strides
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)   # final feature map (B,2048,h',w')
        y = self.head(x)     # (B,1,h',w')
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        return y
