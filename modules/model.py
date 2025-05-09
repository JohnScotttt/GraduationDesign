import torch
import torch.nn as nn
from modules.detail import DetailNet
from modules.ddm import DiffusionNet
from modules.cfg_template import params


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
        out_detail = self.detail_net(low)
        cond = torch.cat((low, gt), dim=1)
        out_diffusion = self.diffusion_net(cond)
        return out_detail, out_diffusion
