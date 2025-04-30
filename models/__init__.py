import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import ShortRangeBranch
from .codec import Decoder, Encoder
from .diffir import ColorRestoration
from .snr import SNRMap
from .transformer import SNRTransformer


class LowLightEnhancement(nn.Module):
    """Low-Light Image Enhancement Model"""

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 transformer_dim=256,
                 time_steps=1000,
                 time_emb_dim=256,
                 branch_weight=[0.5, 0.5]):
        super(LowLightEnhancement, self).__init__()

        # SNR map computation module
        self.snr_module = SNRMap()

        # Encoder
        self.encoder = Encoder(in_channels=in_channels,
                               base_channels=base_channels)

        # Long-range branch
        self.long_range = SNRTransformer(
            dim=transformer_dim,
            patch_size=8,
            depth=6,
            num_heads=8
        )

        # Short-range branch
        self.short_range = ShortRangeBranch(
            in_channels=base_channels*4,
            num_blocks=6
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=base_channels*4,
            base_channels=base_channels,
            out_channels=in_channels
        )

        # Color restoration branch
        self.color_restoration = ColorRestoration(
            in_channels=in_channels,
            base_channels=base_channels,
            time_steps=time_steps,
            time_emb_dim=time_emb_dim
        )

        # Branch fusion weights (fixed hyperparameters)
        self.branch_weight = torch.tensor(branch_weight)

    def forward(self, x):
        # Compute SNR map
        snr_map, snr_mask = self.snr_module(x)

        # Encoder feature extraction
        feat = self.encoder(x)

        # Long-range feature extraction
        long_feat = self.long_range(feat, snr_mask)

        # Short-range feature extraction
        short_feat = self.short_range(feat)

        # Feature fusion
        detail_feat = short_feat * snr_map + long_feat * (1 - snr_map)

        # Detail restoration
        detail_restored = self.decoder(detail_feat)

        # Color restoration
        color_restored = self.color_restoration(x)

        # Final result fusion using fixed weights
        out = self.branch_weight[0] * detail_restored + \
            self.branch_weight[1] * color_restored

        return out, detail_restored, color_restored
