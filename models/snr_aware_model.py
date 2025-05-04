from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def denoise(input_tensor: torch.Tensor) -> torch.Tensor:
    kernel_size = 3
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size),
                        device=input_tensor.device) / (kernel_size * kernel_size)

    return F.conv2d(input_tensor, kernel, padding=padding)


def compute_snr_map(x: torch.Tensor) -> torch.Tensor:
    # 转换为灰度图
    if x.shape[1] == 3:
        # RGB转灰度
        weights = torch.tensor([0.299, 0.587, 0.114],
                               device=x.device).view(1, 3, 1, 1)
        gray = torch.sum(x * weights, dim=1, keepdim=True)
    else:
        gray = x

    # 使用无学习的去噪操作
    denoised = denoise(gray)

    # 估计噪声
    noise = torch.abs(gray - denoised)

    # 计算SNR (Signal-to-Noise Ratio)
    epsilon = 1e-8  # 防止除零
    snr_map = denoised / (noise + epsilon)

    # 归一化SNR图到[0,1]区间
    snr_map = torch.clamp(snr_map / (snr_map.max() + epsilon), 0, 1)

    return snr_map


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SNRGuidedAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SNRGuidedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, snr_mask: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # 生成query, key, value
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, N

        # 应用SNR掩码
        mask_value = -1e9
        snr_mask = snr_mask.unsqueeze(
            1).expand(-1, self.num_heads, -1, -1)  # B, num_heads, N, N
        attn = attn.masked_fill(snr_mask == 0, mask_value)

        # softmax和加权
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SNRGuidedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, snr_mask: torch.Tensor) -> torch.Tensor:
        # 多头自注意力
        x = x + self.attn(self.norm1(x), snr_mask)
        # 前馈网络
        x = x + self.ffn(self.norm2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # 确保x与skip在空间维度上匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SNRAwareTransformer(nn.Module):
    def __init__(self, dim, num_layers=4, num_heads=8, patch_size=16):
        super(SNRAwareTransformer, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_layers = num_layers

        # 特征投影
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 添加线性投影层将patch维度映射为transformer的隐藏维度
        self.patch_to_embedding = nn.Linear(dim * patch_size * patch_size, dim)
        self.embedding_to_patch = nn.Linear(dim, dim * patch_size * patch_size)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, snr_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 确保输入尺寸能够被处理
        # 使用自适应池化调整特征图尺寸为patch_size的倍数
        adaptive_H = (H // self.patch_size) * self.patch_size
        adaptive_W = (W // self.patch_size) * self.patch_size

        # 如果尺寸不匹配，使用自适应池化调整
        if H != adaptive_H or W != adaptive_W:
            x = F.adaptive_avg_pool2d(x, (adaptive_H, adaptive_W))

        # 重新计算patch数量
        h_patches = adaptive_H // self.patch_size
        w_patches = adaptive_W // self.patch_size

        # 如果patch数量太少，使用较小的patch_size
        if h_patches == 0 or w_patches == 0:
            # 调整patch_size
            self.patch_size = max(2, min(adaptive_H, adaptive_W) // 2)
            h_patches = max(1, adaptive_H // self.patch_size)
            w_patches = max(1, adaptive_W // self.patch_size)

        # 将特征图划分为patches
        try:
            x_patches = x.reshape(
                B, C, h_patches, self.patch_size, w_patches, self.patch_size)
            x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            x_patches = x_patches.reshape(
                B, h_patches * w_patches, C * self.patch_size * self.patch_size)
        except RuntimeError:
            # 如果重塑失败，使用更简单的方法：将特征图调整为固定大小然后分块
            x = F.interpolate(
                x,
                size=(h_patches * self.patch_size,
                      w_patches * self.patch_size),
                mode='bilinear',
                align_corners=False
            )
            x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
                3, self.patch_size, self.patch_size)
            x_patches = x_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            x_patches = x_patches.reshape(
                B, h_patches * w_patches, C * self.patch_size * self.patch_size)

        # 将patch投影到transformer维度
        x_embeddings = self.patch_to_embedding(x_patches)

        # 调整SNR map的尺寸以匹配特征图
        # 首先确保SNR map是正确的形状
        snr_map_resized = F.interpolate(
            snr_map,
            size=(h_patches * self.patch_size, w_patches * self.patch_size),
            mode='bilinear',
            align_corners=False
        )

        # 将SNR图划分为patches并计算每个patch的平均SNR
        try:
            snr_patches = snr_map_resized.reshape(
                B, 1, h_patches, self.patch_size, w_patches, self.patch_size)
            snr_patches = snr_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            snr_patches = snr_patches.mean(
                dim=[3, 4, 5])  # B, h_patches, w_patches
        except RuntimeError:
            # 如果重塑失败，使用unfold方法
            snr_patches = snr_map_resized.unfold(2, self.patch_size, self.patch_size).unfold(
                3, self.patch_size, self.patch_size)
            snr_patches = snr_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            snr_patches = snr_patches.mean(
                dim=[3, 4, 5])  # B, h_patches, w_patches

        # 创建SNR掩码矩阵
        snr_mask = snr_patches.reshape(B, -1)  # B, h_patches * w_patches
        threshold = 0.1  # SNR阈值
        snr_mask = (snr_mask >= threshold).float().unsqueeze(
            2)  # B, h_patches * w_patches, 1
        # B, h_patches * w_patches, h_patches * w_patches
        snr_attention_mask = torch.bmm(snr_mask, snr_mask.transpose(1, 2))

        # 应用Transformer层
        for layer in self.transformer_layers:
            x_embeddings = layer(x_embeddings, snr_attention_mask)

        # 将结果投影回原始patch维度
        x_patches = self.embedding_to_patch(x_embeddings)

        # 重构特征图
        try:
            x_patches = x_patches.reshape(
                B, h_patches, w_patches, C, self.patch_size, self.patch_size)
            x_patches = x_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
            x_reconstructed = x_patches.reshape(
                B, C, h_patches * self.patch_size, w_patches * self.patch_size)
        except RuntimeError:
            # 如果重塑失败，使用卷积层重构
            x_patches = x_patches.reshape(
                B * h_patches * w_patches, C, self.patch_size, self.patch_size)
            x_reconstructed = nn.Fold(
                output_size=(h_patches * self.patch_size,
                             w_patches * self.patch_size),
                kernel_size=self.patch_size,
                stride=self.patch_size
            )(x_patches)

        # 将特征图大小调整回原始大小
        if x_reconstructed.shape[2] != H or x_reconstructed.shape[3] != W:
            x_reconstructed = F.interpolate(
                x_reconstructed,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return x_reconstructed


class ShortRangeBranch(nn.Module):
    def __init__(self, channels, num_blocks=6):
        super(ShortRangeBranch, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LowLightEnhancement(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, transformer_dim=256,
                 patch_size=16, num_transformer_layers=4, num_heads=8):
        super(LowLightEnhancement, self).__init__()

        # 编码器 - 3层，步幅分别为1、2、2
        self.enc1 = EncoderBlock(
            in_channels, base_channels, stride=1)  # 保持原始尺寸
        self.enc2 = EncoderBlock(
            base_channels, base_channels * 2, stride=2)  # 1/2
        self.enc3 = EncoderBlock(
            base_channels * 2, transformer_dim, stride=2)  # 1/4

        # 长程分支 - SNR感知Transformer
        self.long_range = SNRAwareTransformer(
            dim=transformer_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            patch_size=patch_size
        )

        # 短程分支 - 卷积残差网络
        self.short_range = ShortRangeBranch(transformer_dim)

        # 解码器 - 2层解码器对应3层编码器
        self.dec3 = DecoderBlock(transformer_dim, base_channels * 2)  # 1/2
        self.dec2 = DecoderBlock(base_channels * 2, base_channels)  # 原始尺寸
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
        )

        # 最终输出层
        self.final = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算SNR图
        snr_map = compute_snr_map(x)

        # 编码器前向传播
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 长程和短程分支处理
        long_range_feature = self.long_range(e3, snr_map)
        short_range_feature = self.short_range(e3)

        # 确保长程特征与e3具有相同的空间尺寸
        if long_range_feature.shape[2:] != e3.shape[2:]:
            long_range_feature = F.interpolate(
                long_range_feature,
                size=e3.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 基于SNR的特征融合
        # 调整SNR图的尺寸以匹配特征
        snr_resized = F.interpolate(
            snr_map, size=e3.shape[2:], mode='bilinear', align_corners=False)
        fusion_weights = snr_resized.mean(dim=[2, 3], keepdim=True)

        # 融合两个分支的特征
        fused_feature = short_range_feature * fusion_weights + \
            long_range_feature * (1 - fusion_weights)

        # 解码器前向传播
        d3 = self.dec3(fused_feature, e2)
        d3 = d3 + e2  # residual connection

        d2 = self.dec2(d3, e1)
        d2 = d2

        d1 = self.dec1(d2)

        # 确保最终输出与输入具有相同的空间尺寸
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(
                d1, size=x.shape[2:], mode='bilinear', align_corners=False)

        # d1作为残差R，添加到原始输入上
        enhanced = x + d1

        # 最终输出
        enhanced = self.final(enhanced)

        return enhanced, snr_map
