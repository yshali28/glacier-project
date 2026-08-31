# ================================================================
# model.py
# Attention U-Net for glacier segmentation
# Input:  3-channel image (NDSI, NDWI, NDVI) — shape (B, 3, H, W)
# Output: binary glacier mask — shape (B, 1, H, W)
# ================================================================

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv -> BatchNorm -> ReLU operations"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate: learns WHERE to focus in the skip connection.

    g = gating signal from deeper (decoder) layer — carries semantic info
    x = skip connection from encoder — carries fine spatial detail

    The gate computes a scalar weight (0 to 1) for each spatial location,
    suppressing irrelevant background features in x.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape != x.shape:
            g = nn.functional.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)
        attention = self.relu(g_proj + x_proj)
        attention = self.psi(attention)
        return x * attention


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary glacier segmentation.

    Architecture:
    - 4-level encoder (progressively doubles channels: 64->128->256->512)
    - Bottleneck at 1024 channels
    - 4-level decoder with Attention Gates on every skip connection
    - Final 1x1 conv + Sigmoid for binary prediction

    Key novelty: Attention Gates guide the decoder to focus on
    glacier-specific spectral signatures in the NDSI/NDWI/NDVI channels.
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,           f * 2)
        self.enc3 = ConvBlock(f * 2,       f * 4)
        self.enc4 = ConvBlock(f * 4,       f * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder: upsampling + attention + convblock
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=f * 8, F_l=f * 8, F_int=f * 4)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3  = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2  = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=f * 2, F_l=f * 2, F_int=f)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1  = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=f, F_l=f, F_int=f // 2)
        self.dec1 = ConvBlock(f * 2, f)

        # Output layer
        self.output = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with Attention
        d4 = self.up4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.output(d1))


# Quick sanity check — run this file directly to verify
if __name__ == "__main__":
    model = AttentionUNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
