import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionStep(nn.Module):
    """Diffusion Step Module"""
    def __init__(self, channels, time_emb_dim=256):
        super(DiffusionStep, self).__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x, t_emb):
        # Time embedding
        t = self.time_mlp(t_emb)
        t = t.view(-1, 1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenate time embedding and input
        x_t = torch.cat([x, t], dim=1)
        
        # Predict noise
        noise_pred = self.noise_predictor(x_t)
        
        return noise_pred

class ColorRestoration(nn.Module):
    """Color Restoration Branch based on DiffIR"""
    def __init__(self, in_channels=3, base_channels=64, time_steps=1000, time_emb_dim=256):
        super(ColorRestoration, self).__init__()
        
        self.time_steps = time_steps
        self.time_emb_dim = time_emb_dim
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        
        # Diffusion steps
        self.diffusion_steps = nn.ModuleList([
            DiffusionStep(base_channels*2, time_emb_dim)
            for _ in range(time_steps)
        ])
        
        # Color reconstruction
        self.color_reconstruction = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Adaptive weight generation
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels*2, 1, 1),
            nn.Sigmoid()
        )
        
    def get_time_embedding(self, t):
        """Generate time embedding"""
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
        
    def forward(self, x, t=None):
        # Feature extraction
        feat = self.feature_extraction(x)
        
        # If time step not provided, sample randomly
        if t is None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=x.device)
        
        # Generate time embedding
        t_emb = self.get_time_embedding(t)
        
        # Diffusion process
        x_t = feat
        for step in range(self.time_steps):
            # Only process samples at current time step
            mask = (t == step)
            if mask.any():
                noise_pred = self.diffusion_steps[step](x_t[mask], t_emb[mask])
                alpha_t = 1 - (step / self.time_steps)
                x_t[mask] = x_t[mask] - alpha_t * noise_pred
        
        # Color reconstruction
        restored = self.color_reconstruction(x_t)
        
        # Generate adaptive weight
        weight = self.weight_generator(x_t)
        
        return restored, weight
