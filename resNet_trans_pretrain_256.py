import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(x)
        x = torch.cat([x, skip], dim=1)

        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


def make_center_prior(H=480, W=640, sigma=0.45):
    y = torch.linspace(-1.0, 1.0, H)
    x = torch.linspace(-1.0, 1.0, W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    prior = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    prior = prior / prior.max()
    return prior.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


class SaliencyResNetTransPretrained(nn.Module):
    """
    ResNet50-ImageNet pretrained encoder + transformer bottleneck + U-Net decoder + learnable center bias.

    Expected input: images in [0, 1], shape (B, 3, H, W).
    The ImageNet normalization is done inside this model.
    """

    def __init__(
        self,
        pretrained=True,
        normalize_input=True,
        freeze_stem=False,
        freeze_layer1=False,
        input_size=(480, 640),
    ):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        # Dilation keeps layer4 at the same spatial size as layer3.
        # For 480x640 input: layer3/layer4 are about 30x40 = 1200 tokens.
        backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, True]
        )

        self.normalize_input = normalize_input
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # Pretrained ResNet encoder
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.layer1  # 256 channels
        self.stage3 = backbone.layer2  # 512 channels
        self.stage4 = backbone.layer3  # 1024 channels
        self.stage5 = backbone.layer4  # 2048 channels, dilated, no extra downsample

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False
        if freeze_layer1:
            for p in self.stage2.parameters():
                p.requires_grad = False

        # Projection layers
        self.bottleneck_proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.skip4_proj = nn.Conv2d(1024, 256, kernel_size=1)
        self.skip3_proj = nn.Conv2d(512, 128, kernel_size=1)
        self.skip2_proj = nn.Conv2d(256, 64, kernel_size=1)
        self.skip1_proj = nn.Conv2d(64, 32, kernel_size=1)

        # Learned positional embedding. For 480x640, bottleneck is 30x40=1200 tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1200, 256))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.Sequential(
            TransformerBlock(dim=256, num_heads=4),
            TransformerBlock(dim=256, num_heads=4),
        )

        # Decoder
        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Learnable center bias initialized as Gaussian prior.
        prior = make_center_prior(input_size[0], input_size[1], sigma=0.45)
        self.center_bias = nn.Parameter(prior)

    def _get_pos_embed(self, Ht, Wt):
        tokens = Ht * Wt
        if self.pos_embed.shape[1] == tokens:
            return self.pos_embed

        # Fallback if input size changes: interpolate positional embedding spatially.
        old_tokens = self.pos_embed.shape[1]
        old_H = int(old_tokens ** 0.5)
        old_W = old_tokens // old_H

        # For 1200 tokens, prefer 30x40.
        if old_tokens == 1200:
            old_H, old_W = 30, 40

        pos = self.pos_embed.transpose(1, 2).reshape(1, 256, old_H, old_W)
        pos = F.interpolate(pos, size=(Ht, Wt), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)
        return pos

    def forward(self, x):
        H, W = x.shape[-2:]

        if self.normalize_input:
            x = (x - self.imagenet_mean) / self.imagenet_std

        # Encoder
        s1 = self.stem(x)           # 64 channels, H/2, W/2
        x = self.maxpool(s1)        # H/4, W/4
        s2 = self.stage2(x)         # 256 channels
        s3 = self.stage3(s2)        # 512 channels
        s4 = self.stage4(s3)        # 1024 channels
        x = self.stage5(s4)         # 2048 channels, same spatial size as s4 due dilation

        # Bottleneck projection
        x = self.bottleneck_proj(x)

        # Transformer bottleneck
        B, C, Ht, Wt = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = x + self._get_pos_embed(Ht, Wt)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, Ht, Wt)

        # Skip projections
        s4 = self.skip4_proj(s4)
        s3 = self.skip3_proj(s3)
        s2 = self.skip2_proj(s2)
        s1 = self.skip1_proj(s1)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        y = self.final(x)

        # Add learnable center bias to logits.
        center_bias = self.center_bias
        if center_bias.shape[-2:] != (H, W):
            center_bias = F.interpolate(center_bias, size=(H, W), mode="bilinear", align_corners=False)
        y = y + center_bias

        return y
