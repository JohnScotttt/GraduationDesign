from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cfg_template import params
from modules.ddm import DiffusionNet
from modules.detail import DetailNet


class LowLightEnhancement(nn.Module):
    def __init__(self, config: params):
        super(LowLightEnhancement, self).__init__()
        self.detail_net = DetailNet(config.detail.in_channels,
                                    config.detail.base_channels,
                                    config.detail.transformer_dim,
                                    config.detail.patch_size,
                                    config.detail.num_transformer_layers,
                                    config.detail.num_heads)
        self.diffusion_net = DiffusionNet(config)

    def forward(self, low: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        detail_out = self.detail_net(low)
        cond = torch.cat((low, gt), dim=1)
        diffusion_out = self.diffusion_net(cond)
        return detail_out, diffusion_out

    def enhance(self, low: torch.Tensor, weight: Tuple[float, float]) -> torch.Tensor:
        detail_out = self.detail_net(low)
        b, c, h, w = low.shape
        img_h_32 = int(32 * np.ceil(h / 32.0))
        img_w_32 = int(32 * np.ceil(w / 32.0))
        low = F.pad(low, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
        diffusion_out = self.diffusion_net(low)["pred_x"]
        diffusion_out = diffusion_out[:, :, :h, :w]
        return weight[0] * detail_out + weight[1] * diffusion_out
