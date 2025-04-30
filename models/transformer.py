import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split feature map into patches"""

    def __init__(self, patch_size):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        # Split feature map into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return x


class SNRMultiHeadAttention(nn.Module):
    """SNR-aware Multi-Head Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(SNRMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, snr_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply SNR mask
        attn = attn + (1 - snr_mask.unsqueeze(1)) * -1e9

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SNRMultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, snr_mask):
        x = x + self.attn(self.norm1(x), snr_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SNRTransformer(nn.Module):
    """SNR-aware Transformer"""

    def __init__(self, dim, patch_size=8, depth=6, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super(SNRTransformer, self).__init__()
        self.patch_embed = PatchEmbed(patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (64//patch_size)**2, dim))
        self.pos_drop = nn.Dropout(p=drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, snr_mask):
        B, C, H, W = x.shape

        # Split feature map into patches
        x = self.patch_embed(x)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, snr_mask)

        x = self.norm(x)

        # Reconstruct feature map
        x = x.view(B, H//8, W//8, C)
        x = x.permute(0, 3, 1, 2)

        return x
