import torch
import torch.nn.functional as F


def _create_window(window_size, sigma, channels):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    gauss_1d = gauss.unsqueeze(1)
    gauss_2d = gauss_1d.mm(gauss_1d.t())

    window = gauss_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    return window


def calculate_psnr(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr.mean().item()


def calculate_ssim(img1, img2, window_size=11, sigma=1.5, L=1.0):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    _, channels, _, _ = img1.shape

    device = img1.device
    window = _create_window(window_size, sigma, channels).to(device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size//2, groups=channels) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()
