import torch
import torch.nn as nn
import torch.nn.functional as F


class SNRMap(nn.Module):
    def __init__(self, kernel_size=3):
        super(SNRMap, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def denoise(self, x):
        """Simple mean filtering for denoising"""
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def compute_snr(self, x):
        """Compute SNR map
        Args:
            x: Input image [B, C, H, W]
        Returns:
            snr_map: SNR map [B, 1, H, W]
        """
        # Convert to grayscale
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # Compute denoised image
        denoised = self.denoise(gray)

        # Compute noise map
        noise = torch.abs(gray - denoised)

        # Compute SNR map (add small value to avoid division by zero)
        snr_map = denoised / (noise + 1e-6)

        return snr_map

    def normalize_snr(self, snr_map):
        """Normalize SNR map to [0,1] range"""
        return (snr_map - snr_map.min()) / (snr_map.max() - snr_map.min() + 1e-6)

    def binarize_snr(self, snr_map, threshold):
        """Binarize SNR map
        Args:
            snr_map: Normalized SNR map
            threshold: Threshold value
        Returns:
            binary_map: Binarized SNR map
        """
        return (snr_map >= threshold).float()

    def forward(self, x, threshold=0.5):
        """Forward pass
        Args:
            x: Input image [B, C, H, W]
            threshold: SNR binarization threshold
        Returns:
            snr_map: Normalized SNR map
            binary_map: Binarized SNR map
        """
        # Compute SNR map
        snr_map = self.compute_snr(x)

        # Normalize
        normalized_snr = self.normalize_snr(snr_map)

        # Binarize
        binary_map = self.binarize_snr(normalized_snr, threshold)

        return normalized_snr, binary_map
