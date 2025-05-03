import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.dataloader import get_dataloader
from models import LowLightEnhancement, DetailLoss
from utils import load_config


# 计算PSNR（峰值信噪比）
def calculate_psnr(img1, img2):
    # 确保图像取值在[0,1]范围内
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # 计算均方误差
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # 计算PSNR
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr.mean().item()


# 创建高斯窗口（用于SSIM计算）
def create_window(window_size, sigma, channels):
    # 创建一维高斯核
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # 创建二维高斯核（外积）
    gauss_1d = gauss.unsqueeze(1)
    gauss_2d = gauss_1d.mm(gauss_1d.t())

    # 扩展到所有通道
    window = gauss_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    return window


# 计算SSIM（结构相似性）
def calculate_ssim(img1, img2, window_size=11, sigma=1.5, L=1.0):
    # 确保图像取值在[0,1]范围内
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # 检查尺寸
    _, channels, _, _ = img1.shape

    # 创建高斯窗口
    device = img1.device
    window = create_window(window_size, sigma, channels).to(device)

    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size//2, groups=channels) - mu1_mu2

    # 稳定性常数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def main():
    # 加载配置文件
    config = load_config('config.yml')

    # 设置训练参数
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    train_tsv_file = config['train_tsv_file']
    val_tsv_file = config['eval_tsv_file']
    learning_rate = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay', 1e-5))

    # 创建输出目录
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader = get_dataloader(train_tsv_file, batch_size=batch_size)
    val_loader = get_dataloader(val_tsv_file, batch_size=batch_size)

    # 初始化模型
    model = LowLightEnhancement(
        in_channels=int(config.get('in_channels', 3)),
        base_channels=int(config.get('base_channels', 64)),
        transformer_dim=int(config.get('transformer_dim', 256)),
        patch_size=int(config.get('patch_size', 16)),
        num_transformer_layers=int(config.get('num_transformer_layers', 4)),
        num_heads=int(config.get('num_heads', 8))
    ).to(device)

    # 设置损失函数
    criterion = DetailLoss().to(device)

    # 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 设置TensorBoard
    writer = SummaryWriter(log_dir=f"{output_dir}/logs")

    # 训练循环
    best_val_loss = float('inf')
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')

        for batch_idx, (low_light, ground_truth) in enumerate(train_pbar):
            low_light = low_light.to(device)
            ground_truth = ground_truth.to(device)
            optimizer.zero_grad()
            enhanced, snr_map = model(low_light)
            loss = criterion(enhanced, ground_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': train_loss / (batch_idx + 1)
            })

            # 定期保存样本图像
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    # 保存一批结果进行可视化
                    sample_idx = 0  # 选择第一个样本
                    sample_low = low_light[sample_idx].cpu()
                    sample_enhanced = enhanced[sample_idx].cpu()
                    sample_gt = ground_truth[sample_idx].cpu()
                    sample_snr = snr_map[sample_idx].cpu()

                    # 创建网格图像
                    grid = torchvision.utils.make_grid([
                        sample_low, sample_enhanced, sample_gt
                    ], nrow=3, normalize=True)

                    writer.add_image(
                        f'samples/train_batch_{batch_idx}', grid, epoch)

                    writer.add_image(f'snr_maps/train_batch_{batch_idx}',
                                     sample_snr, epoch, dataformats='CHW')

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')

        with torch.no_grad():
            for batch_idx, (low_light, ground_truth) in enumerate(val_pbar):
                low_light = low_light.to(device)
                ground_truth = ground_truth.to(device)
                enhanced, snr_map = model(low_light)
                loss = criterion(enhanced, ground_truth)
                psnr = calculate_psnr(enhanced, ground_truth)
                ssim = calculate_ssim(enhanced, ground_truth)
                val_loss += loss.item()
                val_psnr_total += psnr
                val_ssim_total += ssim

                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'psnr': psnr,
                    'ssim': ssim
                })

                # 保存验证样本
                if batch_idx == 0:
                    # 保存一批结果进行可视化
                    sample_idx = 0  # 选择第一个样本
                    sample_low = low_light[sample_idx].cpu()
                    sample_enhanced = enhanced[sample_idx].cpu()
                    sample_gt = ground_truth[sample_idx].cpu()

                    # 创建网格图像
                    grid = torchvision.utils.make_grid([
                        sample_low, sample_enhanced, sample_gt
                    ], nrow=3, normalize=True)

                    writer.add_image('samples/val', grid, epoch)

        val_count = len(val_loader)
        avg_val_loss = val_loss / val_count
        avg_val_psnr = val_psnr_total / val_count
        avg_val_ssim = val_ssim_total / val_count

        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/PSNR', avg_val_psnr, epoch)
        writer.add_scalar('Metrics/SSIM', avg_val_ssim, epoch)
        writer.add_scalar(
            'Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'psnr': avg_val_psnr,
                'ssim': avg_val_ssim,
            }, f"{output_dir}/checkpoints/best_model_loss.pth")
            print(f"保存基于损失的最佳模型, 验证损失: {best_val_loss:.4f}")

        # 保存最佳PSNR模型
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'psnr': best_psnr,
                'ssim': avg_val_ssim,
            }, f"{output_dir}/checkpoints/best_model_psnr.pth")
            print(f"保存基于PSNR的最佳模型, PSNR: {best_psnr:.2f} dB")

        # 保存最佳SSIM模型
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'psnr': avg_val_psnr,
                'ssim': best_ssim,
            }, f"{output_dir}/checkpoints/best_model_ssim.pth")
            print(f"保存基于SSIM的最佳模型, SSIM: {best_ssim:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'psnr': avg_val_psnr,
                'ssim': avg_val_ssim,
            }, f"{output_dir}/checkpoints/epoch_{epoch+1}.pth")

    writer.close()
    print(
        f"训练完成！最佳验证损失: {best_val_loss:.4f}, 最佳PSNR: {best_psnr:.2f} dB, 最佳SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    main()
