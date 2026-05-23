import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Residual Main Path
# =========================================================

class MainPath(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super().__init__()

        F1, F2, F3 = filters

        self.main_path = nn.Sequential(

            nn.Conv2d(
                in_channels,
                F1,
                kernel_size=1,
                stride=stride
            ),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                F1,
                F2,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                F2,
                F3,
                kernel_size=1
            ),
            nn.BatchNorm2d(F3),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Conv2d):

            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.main_path(x)


# =========================================================
# Identity Block
# =========================================================

class IdentityBlock(MainPath):

    def __init__(self, in_channels, filters, kernel_size):

        super().__init__(
            in_channels,
            filters,
            kernel_size
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        y = self.main_path(x) + x

        return self.relu(y)


# =========================================================
# Convolutional Block
# =========================================================

class ConvolutionalBlock(MainPath):

    def __init__(
        self,
        in_channels,
        filters,
        kernel_size,
        stride=2
    ):

        super().__init__(
            in_channels,
            filters,
            kernel_size,
            stride=stride
        )

        F3 = filters[2]

        self.shortcut_path = nn.Sequential(

            nn.Conv2d(
                in_channels,
                F3,
                kernel_size=1,
                stride=stride
            ),

            nn.BatchNorm2d(F3)
        )

        self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def forward(self, x):

        y = self.main_path(x) + self.shortcut_path(x)

        return self.relu(y)


# =========================================================
# U-Net Decoder Block
# =========================================================

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch + skip_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


# =========================================================
# Saliency ResNet U-Net
# =========================================================

class SaliencyResNet(nn.Module):

    def __init__(self, in_channels=3):

        super().__init__()

        # -------------------------------------------------
        # Stem
        # -------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.stage2 = nn.Sequential(

            ConvolutionalBlock(
                64,
                [64, 64, 256],
                kernel_size=3,
                stride=1
            ),

            IdentityBlock(
                256,
                [64, 64, 256],
                kernel_size=3
            ),

            IdentityBlock(
                256,
                [64, 64, 256],
                kernel_size=3
            ),
        )

        self.stage3 = nn.Sequential(

            ConvolutionalBlock(
                256,
                [128, 128, 512],
                kernel_size=3,
                stride=2
            ),

            IdentityBlock(
                512,
                [128, 128, 512],
                kernel_size=3
            ),

            IdentityBlock(
                512,
                [128, 128, 512],
                kernel_size=3
            ),

            IdentityBlock(
                512,
                [128, 128, 512],
                kernel_size=3
            ),
        )

        self.stage4 = nn.Sequential(

            ConvolutionalBlock(
                512,
                [256, 256, 1024],
                kernel_size=3,
                stride=2
            ),

            *[
                IdentityBlock(
                    1024,
                    [256, 256, 1024],
                    kernel_size=3
                )
                for _ in range(5)
            ]
        )

        self.stage5 = nn.Sequential(

            ConvolutionalBlock(
                1024,
                [512, 512, 2048],
                kernel_size=3,
                stride=2
            ),

            IdentityBlock(
                2048,
                [512, 512, 2048],
                kernel_size=3
            ),

            IdentityBlock(
                2048,
                [512, 512, 2048],
                kernel_size=3
            ),
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # -------------------------------------------------
        # Final Prediction
        # -------------------------------------------------

        self.final = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):

        H, W = x.shape[-2:]

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        x = self.stage5(s4)

        # -------------------------------------------------
        # Decoder + Skip Connections
        # -------------------------------------------------

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # -------------------------------------------------
        # Final Resolution
        # -------------------------------------------------

        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        y = self.final(x)

        return y