from torchmetrics.image import PeakSignalNoiseRatio as PNSR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def calculate_psnr(pred, target, device):
    pnsr = PNSR().to(device)
    psnr_value = pnsr(pred, target)
    return psnr_value


def calculate_ssim(pred, target, device):
    ssim = SSIM().to(device)
    ssim_value = ssim(pred, target)
    return ssim_value
