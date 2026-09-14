# -*- coding: utf-8 -*-
"""
model.py — Models for MRI Image Reconstruction
=================================================

Models:
  UNet   — Standard U-Net (Ronneberger et al., MICCAI 2015)
  DnCNN  — Denoising CNN (Zhang et al., IEEE TIP 2017)

Both adapted for MRI denoising: single/dual-channel input → single-channel output.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ===================================================================
# U-Net (Ronneberger et al. 2015)
# ===================================================================
class ConvBlock(nn.Module):
    """Two 3×3 conv layers with BatchNorm, ReLU, and optional Dropout."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net for image-to-image regression.

    Parameters
    ----------
    in_channels : int
        Number of input channels. 1 for single-receiver, 2 for dual.
    out_channels : int
        Number of output channels. Default 1.
    base_features : int
        Features in first encoder layer. Default 32.
        Each level doubles: 32→64→128→256→512.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_features: int = 32, dropout: float = 0.0):
        super().__init__()
        f = base_features
        self.enc1 = ConvBlock(in_channels, f, dropout)
        self.enc2 = ConvBlock(f, f * 2, dropout)
        self.enc3 = ConvBlock(f * 2, f * 4, dropout)
        self.enc4 = ConvBlock(f * 4, f * 8, dropout)
        self.bottleneck = ConvBlock(f * 8, f * 16, dropout)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8, dropout)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, dropout)
        self.out_conv = nn.Conv2d(f, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ===================================================================
# DnCNN (Zhang et al. 2017)
# ===================================================================
class DnCNN(nn.Module):
    """Denoising CNN with direct prediction.

    Unlike the original DnCNN which uses residual learning (noise = model(input)),
    this implementation uses DIRECT prediction (clean = model(input)) because
    residual learning fails at ULF-MRI SNR (~6 dB). At such low SNR, the
    "noise" is larger than the signal, making residual prediction unstable.

    Reference: Zhang et al., "Beyond a Gaussian Denoiser", IEEE TIP 2017.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 1.
    out_channels : int
        Number of output channels. Default 1.
    num_layers : int
        Number of convolutional layers. Default 17 (original paper).
    features : int
        Number of feature channels. Default 64.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 num_layers: int = 17, features: int = 64):
        super().__init__()
        layers = []
        # First layer: Conv + ReLU (no BN)
        layers.append(nn.Conv2d(in_channels, features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # Last layer: Conv (no BN, no ReLU)
        layers.append(nn.Conv2d(features, out_channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Direct prediction mode (not residual learning).
        # Original DnCNN uses residual (output = input - noise) for
        # Gaussian denoising where input ≈ target. In ULF-MRI the
        # noise level is so high (SNR ~6-8 dB) that the noisy input
        # is NOT a good approximation of the target. Direct prediction
        # is more appropriate here.
        return self.net(x)


# ===================================================================
# SimpleCNN — Minimal baseline (no skip connections, no encoder-decoder)
# ===================================================================
class SimpleCNN(nn.Module):
    """Simple stacked-convolution baseline.

    No skip connections, no encoder-decoder, no pooling.
    Just 8 layers of Conv+BN+ReLU. Serves as "lower bound" DL baseline
    to show that architecture matters.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. Default 1.
    features : int
        Feature channels. Default 64.
    num_layers : int
        Number of conv layers. Default 8.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: int = 64, num_layers: int = 8):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, 3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===================================================================
# ResUNet — U-Net with residual blocks (common MRI reconstruction variant)
# ===================================================================
class ResBlock(nn.Module):
    """Residual block: Conv+BN+ReLU+Conv+BN + skip."""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        ]
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResUNet(nn.Module):
    """U-Net with residual blocks instead of plain ConvBlocks.

    Reference: Adapted from Lee et al., "Deeply-Supervised Nets with
    Residual Learning for Brain Tumor Segmentation", BrainLes 2017.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    base_features : int
        Default 32.
    dropout : float
        Default 0.0.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_features: int = 32, dropout: float = 0.0):
        super().__init__()
        f = base_features

        # Encoder: initial conv + residual blocks
        self.enc1_init = nn.Sequential(
            nn.Conv2d(in_channels, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f), nn.ReLU(inplace=True))
        self.enc1 = ResBlock(f, dropout)
        self.enc2_init = nn.Sequential(
            nn.Conv2d(f, f*2, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(f*2), nn.ReLU(inplace=True))
        self.enc2 = ResBlock(f*2, dropout)
        self.enc3_init = nn.Sequential(
            nn.Conv2d(f*2, f*4, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(f*4), nn.ReLU(inplace=True))
        self.enc3 = ResBlock(f*4, dropout)
        self.enc4_init = nn.Sequential(
            nn.Conv2d(f*4, f*8, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(f*8), nn.ReLU(inplace=True))
        self.enc4 = ResBlock(f*8, dropout)

        # Bottleneck
        self.bottleneck_init = nn.Sequential(
            nn.Conv2d(f*8, f*16, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(f*16), nn.ReLU(inplace=True))
        self.bottleneck = ResBlock(f*16, dropout)

        # Decoder: upsample → cat skip → 1x1 reduce → ResBlock
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.dec4_reduce = nn.Conv2d(f*16, f*8, 1)  # cat f*8+f*8=f*16 → f*8
        self.dec4 = ResBlock(f*8, dropout)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.dec3_reduce = nn.Conv2d(f*8, f*4, 1)
        self.dec3 = ResBlock(f*4, dropout)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.dec2_reduce = nn.Conv2d(f*4, f*2, 1)
        self.dec2 = ResBlock(f*2, dropout)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1_reduce = nn.Conv2d(f*2, f, 1)
        self.dec1 = ResBlock(f, dropout)

        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(self.enc1_init(x))
        e2 = self.enc2(self.enc2_init(e1))
        e3 = self.enc3(self.enc3_init(e2))
        e4 = self.enc4(self.enc4_init(e3))
        b = self.bottleneck(self.bottleneck_init(e4))

        d4 = self.dec4(self.dec4_reduce(torch.cat([self.up4(b), e4], dim=1)))
        d3 = self.dec3(self.dec3_reduce(torch.cat([self.up3(d4), e3], dim=1)))
        d2 = self.dec2(self.dec2_reduce(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1(self.dec1_reduce(torch.cat([self.up1(d2), e1], dim=1)))

        return self.out_conv(d1)


# ===================================================================
# NAFNet (Chen et al., ECCV 2022)
# ===================================================================
class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Normalize in float32 to prevent fp16 overflow (matches PyTorch LayerNorm)
        dtype = x.dtype
        x = x.float()
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, correction=0)
        x = ((x - mean) / (var + self.eps).sqrt()
             * self.weight[None, :, None, None].float()
             + self.bias[None, :, None, None].float())
        return x.to(dtype)


class SimpleGate(nn.Module):
    """Gated Linear Unit without activation: split channels, multiply."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core building block of NAFNet.

    Two sub-blocks with learnable residual scaling (beta, gamma):
    1. Spatial mixing: LayerNorm → 1×1 → DWConv3×3 → SimpleGate → SCA → 1×1
    2. Channel mixing: LayerNorm → 1×1 → SimpleGate → 1×1

    No activation functions anywhere. SimpleGate provides nonlinearity.
    Reference: Chen et al., "Simple Baselines for Image Restoration," ECCV 2022.
    """
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw_ch = channels * dw_expand
        ffn_ch = channels * ffn_expand

        # Sub-block 1: spatial mixing
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, 1, bias=True)
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.gate1 = SimpleGate()  # dw_ch → dw_ch//2 = channels
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=True)

        # Sub-block 2: channel mixing (FFN)
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.gate2 = SimpleGate()  # ffn_ch → ffn_ch//2 = channels
        self.conv4 = nn.Conv2d(channels, channels, 1, bias=True)

        # Learnable residual scaling (init to zero → identity at start)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Sub-block 1: spatial
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dw_conv(y)
        y = self.gate1(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        y = self.dropout(y)
        x = x + y * self.beta

        # Sub-block 2: channel (FFN)
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.gate2(y)
        y = self.conv4(y)
        y = self.dropout(y)
        x = x + y * self.gamma

        return x


class NAFNet(nn.Module):
    """Nonlinear Activation Free Network for image restoration.

    U-Net-style encoder-decoder with NAFBlocks. No activation functions —
    all nonlinearity comes from SimpleGate (channel splitting + multiplication).

    Direct prediction (not residual learning) — matches ULF-MRI SNR regime
    where residual learning fails.

    Reference: Chen et al., "Simple Baselines for Image Restoration," ECCV 2022.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for single sensor, 2 for fusion).
    out_channels : int
        Output channels. Default 1.
    base_features : int
        Base channel width. Default 32.
    enc_blk_nums : list of int
        Number of NAFBlocks per encoder level. Default [2, 2, 4, 8].
    middle_blk_num : int
        Number of NAFBlocks at bottleneck. Default 12.
    dec_blk_nums : list of int
        Number of NAFBlocks per decoder level. Default [2, 2, 2, 2].
    dropout : float
        Dropout rate. Default 0.0.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_features: int = 32, dropout: float = 0.0,
                 enc_blk_nums=None, middle_blk_num: int = 12,
                 dec_blk_nums=None):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.intro = nn.Conv2d(in_channels, base_features, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(base_features, out_channels, 3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_features
        for num_blk in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blk)])
            )
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2

        # Bottleneck
        self.middle = nn.Sequential(
            *[NAFBlock(ch, dropout=dropout) for _ in range(middle_blk_num)]
        )

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num_blk in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 1, bias=False),
                    nn.PixelShuffle(2),  # ch*2 → ch//2 spatially upsampled
                )
            )
            ch //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blk)])
            )

    def forward(self, x):
        x = self.intro(x)

        # Encoder
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle(x)

        # Decoder (skip connections via addition)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        return self.ending(x)


# ===================================================================
# CrossFusionNet — Dual-Encoder Cross-Attention Fusion
# ===================================================================
class SpatialCrossAttention(nn.Module):
    """Bidirectional spatial cross-attention between two feature maps.

    Each pixel in branch A queries ALL pixels in branch B to find
    complementary information, and vice versa.

    For ULF-MRI: SQUID features query Coil features (and reverse),
    enabling per-pixel, per-stage selective fusion.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_a = LayerNorm2d(channels)
        self.norm_b = LayerNorm2d(channels)

        # A queries B
        self.q_a = nn.Conv2d(channels, channels, 1, bias=True)
        self.kv_b = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.proj_a = nn.Conv2d(channels, channels, 1, bias=True)

        # B queries A
        self.q_b = nn.Conv2d(channels, channels, 1, bias=True)
        self.kv_a = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.proj_b = nn.Conv2d(channels, channels, 1, bias=True)

        # Learnable gate (init 0 → start as identity)
        self.gate_a = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate_b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _cross_attn(self, q, kv, proj):
        B, C, H, W = q.shape
        nh, hd = self.num_heads, self.head_dim

        q = q.reshape(B, nh, hd, H * W).permute(0, 1, 3, 2)   # (B, nh, HW, hd)
        k, v = kv.chunk(2, dim=1)
        k = k.reshape(B, nh, hd, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, nh, hd, H * W).permute(0, 1, 3, 2)

        # FlashAttention (PyTorch >= 2.0): O(N) memory, much faster
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return proj(out)

    def forward(self, feat_a, feat_b):
        # A attends to B
        a_norm = self.norm_a(feat_a)
        b_norm = self.norm_b(feat_b)
        cross_a = self._cross_attn(self.q_a(a_norm), self.kv_b(b_norm), self.proj_a)
        feat_a = feat_a + cross_a * self.gate_a

        # B attends to A
        cross_b = self._cross_attn(self.q_b(b_norm), self.kv_a(a_norm), self.proj_b)
        feat_b = feat_b + cross_b * self.gate_b

        return feat_a, feat_b


class CrossFusionNet(nn.Module):
    """Dual-encoder network with cross-attention fusion for multi-sensor MRI.

    Architecture:
      SQUID input → Encoder_S ──┐
                                ├─ Cross-Attention (bidirectional, per level)
      Coil input  → Encoder_C ──┘
                                │
                        Concatenate bottleneck
                                │
                          Shared Decoder
                                │
                            Output (1ch)

    Each encoder level processes its sensor independently, then
    cross-attention lets each pixel query the other sensor's features
    to extract complementary information.

    Uses NAFBlock as the building block (no activation functions).

    Parameters
    ----------
    base_features : int
        Base channel width. Default 16.
    enc_blk_nums : list of int
        NAFBlocks per encoder level. Default [2, 2, 4].
    middle_blk_num : int
        NAFBlocks at bottleneck. Default 4.
    dec_blk_nums : list of int
        NAFBlocks per decoder level. Default [2, 2, 2].
    num_heads : int
        Attention heads for cross-attention. Default 4.
    dropout : float
        Dropout rate. Default 0.0.
    """
    def __init__(self, base_features: int = 24, dropout: float = 0.0,
                 enc_blk_nums=None, middle_blk_num: int = 4,
                 dec_blk_nums=None, num_heads: int = 4,
                 cross_attn_start_level: int = 2):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 4]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]
        self.cross_attn_start_level = cross_attn_start_level

        # Separate intro for each sensor (1ch → base_features)
        self.intro_s = nn.Conv2d(1, base_features, 3, padding=1, bias=True)
        self.intro_c = nn.Conv2d(1, base_features, 3, padding=1, bias=True)

        # Dual encoders + cross-attention at each level
        self.enc_s = nn.ModuleList()
        self.enc_c = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.downs_s = nn.ModuleList()
        self.downs_c = nn.ModuleList()

        ch = base_features
        for level, num_blk in enumerate(enc_blk_nums):
            self.enc_s.append(nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blk)]))
            self.enc_c.append(nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blk)]))
            # Cross-attention only at lower resolutions (saves memory, global context)
            if level >= cross_attn_start_level:
                self.cross_attn.append(SpatialCrossAttention(ch, num_heads=max(1, ch // 16)))
            else:
                self.cross_attn.append(nn.Identity())  # placeholder at high resolution
            self.downs_s.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            self.downs_c.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2

        # Bottleneck: merge both branches (concatenate → reduce)
        self.bottleneck_reduce = nn.Conv2d(ch * 2, ch, 1, bias=True)
        self.middle = nn.Sequential(
            *[NAFBlock(ch, dropout=dropout) for _ in range(middle_blk_num)]
        )

        # Shared decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num_blk in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blk)])
            )

        self.ending = nn.Conv2d(ch, 1, 3, padding=1, bias=True)

    def forward(self, x):
        # Split 2-channel input into SQUID (ch0) and Coil (ch1)
        assert x.shape[1] == 2, f"CrossFusionNet requires 2-channel input, got {x.shape[1]}"
        x_s = self.intro_s(x[:, 0:1])
        x_c = self.intro_c(x[:, 1:2])

        # Dual encoder with cross-attention
        skips_s = []
        skips_c = []
        for enc_s, enc_c, cross, down_s, down_c in zip(
                self.enc_s, self.enc_c, self.cross_attn,
                self.downs_s, self.downs_c):
            x_s = enc_s(x_s)
            x_c = enc_c(x_c)
            if isinstance(cross, SpatialCrossAttention):
                x_s, x_c = cross(x_s, x_c)  # bidirectional cross-attention
            skips_s.append(x_s)
            skips_c.append(x_c)
            x_s = down_s(x_s)
            x_c = down_c(x_c)

        # Merge at bottleneck
        x = self.bottleneck_reduce(torch.cat([x_s, x_c], dim=1))
        x = self.middle(x)

        # Shared decoder (skip = sum of both encoder skips)
        for dec, up, sk_s, sk_c in zip(
                self.decoders, self.ups,
                reversed(skips_s), reversed(skips_c)):
            x = up(x)
            x = x + sk_s + sk_c  # combine both skips
            x = dec(x)

        return self.ending(x)


# ===================================================================
# PoE-VAE — Product-of-Experts Latent Fusion
# ===================================================================
class PoEFusionVAE(nn.Module):
    """Dual-encoder VAE with Product-of-Experts latent fusion.

    Each sensor encodes to a Gaussian (μ, σ²) in spatial latent space.
    PoE fuses them: the fused precision = sum of individual precisions.
    This is the Bayesian-optimal fusion of independent Gaussian beliefs.

    Key property: σ²_fused ≤ min(σ²_s, σ²_c) — information only increases.
    The σ² maps reveal WHERE each sensor is confident.

    Training: MSE(recon, ideal) + β × KL(q_fused || N(0,1))
    Inference: z = μ_fused (deterministic, no sampling)

    Reference: Wu & Goodman, "Multimodal Generative Models for Scalable
    Weakly-Supervised Learning," NeurIPS 2018.

    Parameters
    ----------
    base_ch : int
        Base channels. Default 48. Encoder: C→2C→4C→8C (=384 at bottleneck).
        Gives ~10M params total, comparable to CrossFusionNet (9.48M).
    latent_ch : int
        Latent channels (per μ and logvar). Default 8.
        Spatial latent: 8 × 16×16 = 2048 dims.
    n_blocks : int
        NAFBlocks per encoder/decoder level. Default 2.
    dropout : float
        Dropout rate. Default 0.0.
    """
    def __init__(self, base_ch: int = 48, latent_ch: int = 8,
                 n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.latent_ch = latent_ch
        C = base_ch

        # Encoder (per sensor, separate weights)
        # Channel progression: C → 2C → 4C → 8C (proper doubling at each level)
        def make_encoder():
            layers = nn.ModuleList()
            # Level 0: 128×128, C channels
            layers.append(nn.Conv2d(1, C, 3, padding=1))
            layers.append(nn.Sequential(*[NAFBlock(C, dropout=dropout) for _ in range(n_blocks)]))
            # Level 1: 64×64, 2C channels
            layers.append(nn.Conv2d(C, C*2, 2, stride=2))
            layers.append(nn.Sequential(*[NAFBlock(C*2, dropout=dropout) for _ in range(n_blocks)]))
            # Level 2: 32×32, 4C channels
            layers.append(nn.Conv2d(C*2, C*4, 2, stride=2))
            layers.append(nn.Sequential(*[NAFBlock(C*4, dropout=dropout) for _ in range(n_blocks)]))
            # Level 3: 16×16, 8C channels → project to 2×latent_ch (μ and logvar)
            layers.append(nn.Conv2d(C*4, C*8, 2, stride=2))
            layers.append(nn.Sequential(*[NAFBlock(C*8, dropout=dropout) for _ in range(n_blocks)]))
            layers.append(nn.Conv2d(C*8, latent_ch * 2, 1))
            return layers

        self.enc_s = make_encoder()
        self.enc_c = make_encoder()

        # Decoder (shared, mirrors encoder)
        self.dec = nn.ModuleList()
        # Level 3: 16×16, 8C
        self.dec.append(nn.Conv2d(latent_ch, C*8, 1))
        self.dec.append(nn.Sequential(*[NAFBlock(C*8, dropout=dropout) for _ in range(n_blocks)]))
        # Level 2: 32×32, 4C
        self.dec.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       nn.Conv2d(C*8, C*4, 3, padding=1)))
        self.dec.append(nn.Sequential(*[NAFBlock(C*4, dropout=dropout) for _ in range(n_blocks)]))
        # Level 1: 64×64, 2C
        self.dec.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       nn.Conv2d(C*4, C*2, 3, padding=1)))
        self.dec.append(nn.Sequential(*[NAFBlock(C*2, dropout=dropout) for _ in range(n_blocks)]))
        # Level 0: 128×128, C → 1ch
        self.dec.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       nn.Conv2d(C*2, C, 3, padding=1)))
        self.dec.append(nn.Sequential(*[NAFBlock(C, dropout=dropout) for _ in range(n_blocks)]))
        self.dec.append(nn.Conv2d(C, 1, 3, padding=1))

    def _encode(self, x, encoder):
        """Run one encoder branch, return (μ, logvar) in float32."""
        for layer in encoder:
            x = layer(x)
        # Cast to float32 before splitting — prevents fp16 overflow in exp()
        x = x.float()
        mu, logvar = x.chunk(2, dim=1)  # split latent_ch*2 → latent_ch + latent_ch
        logvar = logvar.clamp(-10, 10)  # numerical stability (tighter for safety)
        return mu, logvar

    def _decode(self, z):
        """Decode latent z to image."""
        x = z
        for layer in self.dec:
            x = layer(x)
        return x

    @staticmethod
    def poe_fuse(mu_s, logvar_s, mu_c, logvar_c):
        """Product of Experts fusion of two Gaussians + N(0,1) prior.

        All computation in float32 to prevent overflow.
        Returns fused (μ, logvar) with higher precision than either input.
        """
        # Ensure float32 (inputs from _encode are already float32,
        # but be defensive)
        mu_s, logvar_s = mu_s.float(), logvar_s.float()
        mu_c, logvar_c = mu_c.float(), logvar_c.float()

        T_s = torch.exp(-logvar_s)      # SQUID precision
        T_c = torch.exp(-logvar_c)      # Coil precision
        T_prior = 1.0                   # N(0,1) prior precision

        T_fused = T_s + T_c + T_prior
        mu_fused = (T_s * mu_s + T_c * mu_c) / T_fused  # prior μ=0
        logvar_fused = -torch.log(T_fused)

        return mu_fused, logvar_fused

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick: z = μ + σ × ε, ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(mu, logvar):
        """KL(N(μ, σ²) || N(0, 1)) = -0.5 × Σ(1 + logσ² - μ² - σ²)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x, deterministic=False):
        """
        x: (B, 2, H, W) — stacked [SQUID, Coil]
        deterministic: if True, z = μ (no sampling, for inference)
        Returns: (recon, mu_fused, logvar_fused)
        """
        squid = x[:, 0:1]
        coil = x[:, 1:2]

        # Encode each sensor
        mu_s, logvar_s = self._encode(squid, self.enc_s)
        mu_c, logvar_c = self._encode(coil, self.enc_c)

        # PoE fusion
        mu_f, logvar_f = self.poe_fuse(mu_s, logvar_s, mu_c, logvar_c)

        # Sample or use mean
        if deterministic or not self.training:
            z = mu_f
        else:
            z = self.reparameterize(mu_f, logvar_f)

        # Decode
        recon = self._decode(z)

        return recon, mu_f, logvar_f


if __name__ == "__main__":
    for Model, name in [(UNet, "UNet"), (DnCNN, "DnCNN"),
                         (SimpleCNN, "SimpleCNN"), (ResUNet, "ResUNet"),
                         (NAFNet, "NAFNet")]:
        for in_ch in [1, 2]:
            model = Model(in_channels=in_ch)
            x = torch.randn(2, in_ch, 128, 128)
            y = model(x)
            n = sum(p.numel() for p in model.parameters())
            print(f"  {name}(in={in_ch}): {x.shape} → {y.shape}, {n/1e6:.2f}M params")
    # CrossFusionNet (2-channel only)
    cf = CrossFusionNet()
    x2 = torch.randn(2, 2, 128, 128)
    y2 = cf(x2)
    n2 = sum(p.numel() for p in cf.parameters())
    print(f"  CrossFusionNet(in=2): {x2.shape} → {y2.shape}, {n2/1e6:.2f}M params")
    # PoEFusionVAE (2-channel only)
    poe = PoEFusionVAE()
    x_poe = torch.randn(2, 2, 128, 128)
    recon, mu_f, logvar_f = poe(x_poe)
    n_poe = sum(p.numel() for p in poe.parameters())
    kl = PoEFusionVAE.kl_divergence(mu_f, logvar_f)
    print(f"  PoEFusionVAE(in=2): {x_poe.shape} → recon{recon.shape}, "
          f"latent{mu_f.shape}, KL={kl.item():.4f}, {n_poe/1e6:.2f}M params")
    # Verify PoE property: fused variance < both individual variances
    poe.eval()
    with torch.no_grad():
        mu_s, lv_s = poe._encode(x_poe[:, 0:1], poe.enc_s)
        mu_c, lv_c = poe._encode(x_poe[:, 1:2], poe.enc_c)
        _, lv_f = PoEFusionVAE.poe_fuse(mu_s, lv_s, mu_c, lv_c)
        var_s = lv_s.exp().mean().item()
        var_c = lv_c.exp().mean().item()
        var_f = lv_f.exp().mean().item()
        assert var_f < min(var_s, var_c), f"PoE violation: {var_f} not < min({var_s}, {var_c})"
    print(f"    σ²: SQUID={var_s:.4f}, Coil={var_c:.4f}, Fused={var_f:.4f} (fused < both ✓)")
    print("  OK ✓")
