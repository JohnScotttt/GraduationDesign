from torchmetrics.image import PeakSignalNoiseRatio as PNSR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torch

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> float:
    pnsr = PNSR().to(device)
    psnr_value = pnsr(pred, target)
    return psnr_value.item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> float:
    ssim = SSIM().to(device)
    ssim_value = ssim(pred, target)
    return ssim_value.item()
