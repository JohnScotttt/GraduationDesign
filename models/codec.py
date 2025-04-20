import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder Module"""
    def __init__(self, in_channels=3, base_channels=64):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # First layer
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Second layer
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            # Third layer
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """Decoder Module"""
    def __init__(self, in_channels=256, base_channels=64, out_channels=3):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            # First upsampling layer
            nn.ConvTranspose2d(in_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            # Second upsampling layer
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(x)
