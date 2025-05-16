import os
import sys

import lpips
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from modules.cfg_template import params
from modules.dataloader import get_dataloader
from modules.ddm import EMAHelper
from modules.loss import LowLightLoss
from modules.model import LowLightEnhancement
from utils import load_config
from utils.metrics import calculate_psnr, calculate_ssim


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, score):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

def train(config_file: str = None):
    default_cfg = load_config('cfg/default.toml')
    config = load_config(config_file) if config_file else {}

    default_cfg.update(config)
    cfg = params(**default_cfg)

    if cfg.settings.device == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA is not available, using CPU instead')
            device = torch.device('cpu')
        else:
            print('CUDA is available, using GPU')
            device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    os.makedirs(cfg.settings.output_dir, exist_ok=True)
    suffix = 1
    name = cfg.settings.name
    while os.path.exists(os.path.join(cfg.settings.output_dir, name)):
        name = f"{cfg.settings.name}_{suffix}"
        suffix += 1
    del suffix
    os.makedirs(os.path.join(cfg.settings.output_dir, name), exist_ok=True)
    ckpt_dir = os.path.join(cfg.settings.output_dir, name, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(cfg.settings.output_dir, name, 'log')
    os.makedirs(log_dir, exist_ok=True)

    train_loader = get_dataloader(cfg.settings.train_tsv_file,
                                  batch_size=cfg.settings.batch_size,
                                  num_workers=cfg.settings.num_workers,
                                  patch_size=cfg.settings.patch_size)
    val_loader = get_dataloader(cfg.settings.eval_tsv_file,
                                batch_size=cfg.settings.batch_size,
                                num_workers=cfg.settings.num_workers)

    model = LowLightEnhancement(cfg)
    model.to(device)
    if cfg.settings.ckpt != '':
        model.load_state_dict(torch.load(cfg.settings.ckpt))

    ema_helper = EMAHelper()
    ema_helper.register(model)

    if cfg.settings.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.settings.lr, weight_decay=cfg.settings.weight_decay)
    elif cfg.settings.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.settings.lr, momentum=cfg.settings.momentum)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.settings.optimizer}")

    if cfg.settings.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.settings.step_size, gamma=cfg.settings.factor)
    elif cfg.settings.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=cfg.settings.factor,
            patience=cfg.settings.patience, min_lr=cfg.settings.min_lr)
    else:
        raise ValueError(f"Invalid scheduler: {cfg.settings.scheduler}")

    criterion = LowLightLoss(cfg.settings.weight).to(device)

    if cfg.settings.early_stop > 0:
        early_stopping = EarlyStopping(patience=cfg.settings.early_stop)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    best_score = 0.0

    epoch_iterator = tqdm(range(cfg.settings.epochs), desc=f"Training ({name})", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_iterator:
        cudnn.benchmark = True

        # train
        model.train()
        epoch_total_loss = 0.0
        epoch_detail_loss = 0.0
        epoch_noise_loss = 0.0
        epoch_photo_loss = 0.0
        epoch_frequency_loss = 0.0

        train_batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.settings.epochs} Training",
                                    unit="batch", leave=False, dynamic_ncols=True)

        for batch_idx, (low, gt) in enumerate(train_batch_iterator):
            low = low.to(device)
            gt = gt.to(device)

            detail_output, diffusion_output = model(low, gt)
            total_loss, detail_loss, noise_loss, photo_loss, frequency_loss = criterion(
                (detail_output, diffusion_output), gt)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            ema_helper.update(model)

            # Accumulate losses for the epoch
            epoch_total_loss += total_loss.item()
            epoch_detail_loss += detail_loss.item()
            epoch_noise_loss += noise_loss.item()
            epoch_photo_loss += photo_loss.item()
            epoch_frequency_loss += frequency_loss.item()

        train_batch_iterator.close()

        # Log training losses to TensorBoard
        writer.add_scalar('Loss/Total', epoch_total_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Detail', epoch_detail_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Noise', epoch_noise_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Photo', epoch_photo_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Frequency', epoch_frequency_loss / len(train_loader), epoch)

        # eval
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        val_batch_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.settings.epochs} Validation",
                                  unit="batch", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, (low, gt) in enumerate(val_batch_iterator):
                low = low.to(device)
                gt = gt.to(device)

                enhance_img = model.enhance(low, cfg.settings.weight)
                total_psnr += calculate_psnr(enhance_img, gt, device)
                total_ssim += calculate_ssim(enhance_img, gt, device)

            val_batch_iterator.close()

        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        avg_score = 0.5 * (avg_psnr / 50) + 0.5 * avg_ssim

        # Log validation metrics to TensorBoard
        writer.add_scalar('Validation/PSNR', avg_psnr, epoch)
        writer.add_scalar('Validation/SSIM', avg_ssim, epoch)
        writer.add_scalar('Validation/Score', avg_score, epoch)

        if avg_score > best_score:
            best_score = avg_score
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'best_model_{avg_psnr}_{avg_ssim}.pth'))

        if early_stopping(avg_score):
            break

        if cfg.settings.scheduler == 'StepLR':
            scheduler.step()
        elif cfg.settings.scheduler == 'ReduceLROnPlateau':
            scheduler.step(avg_score)

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'last_model.pth'))
        
    epoch_iterator.close()

    # Close the TensorBoard writer
    writer.close()


def eval(config_file: str = None, model_path: str = None):
    """
    评估模型在验证集上的性能
    
    Args:
        config_file (str): 配置文件路径
        model_path (str): 模型权重文件路径
    """
    if not model_path:
        raise ValueError("必须提供模型权重文件路径")
        
    default_cfg = load_config('cfg/default.toml')
    config = load_config(config_file) if config_file else {}
    default_cfg.update(config)
    cfg = params(**default_cfg)
    
    # 设置设备
    if cfg.settings.device == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA不可用，使用CPU')
            device = torch.device('cpu')
        else:
            print('使用GPU')
            device = torch.device('cuda')
    else:
        print('使用CPU')
        device = torch.device('cpu')
    
    # 加载数据集
    val_loader = get_dataloader(tsv_file=cfg.settings.train_tsv_file,
                                batch_size=cfg.settings.batch_size,
                                shuffle=False,
                                num_workers=cfg.settings.num_workers,
                                patch_size=cfg.settings.patch_size,
                                )
    
    # 初始化模型
    model = LowLightEnhancement(cfg)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 初始化LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # 初始化评估指标
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    
    # 使用tqdm显示进度
    eval_iterator = tqdm(val_loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_idx, (low, gt) in enumerate(eval_iterator):
            low = low.to(device)
            gt = gt.to(device)
            
            # 获取增强后的图像
            enhance_img = model.enhance(low, cfg.settings.weight)
            
            # 计算评估指标（修正批处理计算逻辑）
            batch_size = low.size(0)
            batch_psnr = calculate_psnr(enhance_img, gt, device)
            batch_ssim = calculate_ssim(enhance_img, gt, device)
            batch_lpips = lpips_fn(enhance_img, gt).mean().item()
            
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_lpips += batch_lpips * batch_size
            
            # 更新进度条
            eval_iterator.set_postfix({
                'PSNR': f'{batch_psnr:.4f}',
                'SSIM': f'{batch_ssim:.4f}',
                'LPIPS': f'{batch_lpips:.4f}'
            })
    
    # 计算平均值
    num_samples = len(val_loader.dataset) # 使用数据集的总样本数
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    avg_lpips = total_lpips / num_samples if num_samples > 0 else 0
    
    print("\n评估结果:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }