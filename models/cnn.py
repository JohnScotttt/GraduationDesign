import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ShortRangeBranch(nn.Module):
    """Short-Range Branch"""
    def __init__(self, in_channels=64, num_blocks=6):
        super(ShortRangeBranch, self).__init__()
        
        # Initial convolution layer
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels) for _ in range(num_blocks)
        ])
        
        # Output convolution layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Initial feature extraction
        x = self.conv_in(x)
        
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Output features
        x = self.conv_out(x)
        
        return x
