import os
import sys

import torch
from tqdm import tqdm

from modules.cfg_template import params
from modules.dataloader import get_dataloader
from modules.loss import DetailSimpleLoss, DiffusionLoss
from modules.model import LowLightEnhancement
from utils import load_config
from utils.metrics import calculate_psnr, calculate_ssim

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def train(config_file: str):
    default_cfg = load_config('cfg/default.toml')
    config = load_config(config_file)

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
    os.makedirs(os.path.join(cfg.settings.output_dir, name, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(cfg.settings.output_dir, name, 'log'), exist_ok=True)

    train_loader = get_dataloader(
        cfg.settings.train_tsv_file, batch_size=cfg.settings.batch_size, num_workers=cfg.settings.num_workers)
    val_loader = get_dataloader(
        cfg.settings.eval_tsv_file, batch_size=cfg.settings.batch_size, num_workers=cfg.settings.num_workers)

    model = LowLightEnhancement(cfg)
    model.to(device)
    if cfg.settings.ckpt != '':
        model.load_state_dict(torch.load(cfg.settings.ckpt))

    # Detail
    if cfg.detail.optimizer == 'Adam':
        detail_optimizer = torch.optim.Adam(
            model.detail_net.parameters(), lr=cfg.detail.lr, weight_decay=cfg.detail.weight_decay)
    elif cfg.detail.optimizer == 'SGD':
        detail_optimizer = torch.optim.SGD(
            model.detail_net.parameters(), lr=cfg.detail.lr, momentum=cfg.detail.momentum)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.detail.optimizer}")

    if cfg.detail.scheduler == 'StepLR':
        detail_scheduler = torch.optim.lr_scheduler.StepLR(
            detail_optimizer, step_size=cfg.detail.step_size, gamma=cfg.detail.factor)
    elif cfg.detail.scheduler == 'ReduceLROnPlateau':
        detail_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            detail_optimizer, factor=cfg.detail.factor, patience=cfg.detail.patience, min_lr=cfg.detail.min_lr)
    else:
        raise ValueError(f"Invalid scheduler: {cfg.detail.scheduler}")

    detail_criterion = DetailSimpleLoss().to(device)

    # Diffusion
    if cfg.diffusion.optimizer == 'Adam':
        diffusion_optimizer = torch.optim.Adam(
            model.diffusion_net.parameters(), lr=cfg.diffusion.lr, weight_decay=cfg.diffusion.weight_decay,
            betas=(0.9, 0.999), amsgrad=cfg.diffusion.amsgrad, eps=cfg.diffusion.eps)
    elif cfg.diffusion.optimizer == 'SGD':
        diffusion_optimizer = torch.optim.SGD(
            model.diffusion_net.parameters(), lr=cfg.diffusion.lr, momentum=cfg.diffusion.momentum)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.diffusion.optimizer}")

    if cfg.diffusion.scheduler == 'StepLR':
        diffusion_scheduler = torch.optim.lr_scheduler.StepLR(
            diffusion_optimizer, step_size=cfg.diffusion.step_size, gamma=cfg.diffusion.factor)
    elif cfg.diffusion.scheduler == 'ReduceLROnPlateau':
        diffusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            diffusion_optimizer, factor=cfg.diffusion.factor, patience=cfg.diffusion.patience, min_lr=cfg.diffusion.min_lr)
    else:
        raise ValueError(f"Invalid scheduler: {cfg.diffusion.scheduler}")

    diffusion_criterion = DiffusionLoss().to(device)

    # 训练循环
    best_psnr = 0.0
    best_ssim = 0.0
    best_loss = float('inf')

    if cfg.settings.early_stop > 0:
        early_stopping = EarlyStopping(patience=cfg.settings.early_stop)

    for epoch in range(cfg.settings.epochs):
        # train
        for batch_idx, (low, gt) in enumerate(train_loader):
            model.train()
            train_loss = 0.0
            low = low.to(device)
            gt = gt.to(device)

            detail_output, diffusion_output = model(low, gt)
            loss = detail_criterion(detail_output, gt) + diffusion_criterion(diffusion_output, gt)
            train_loss += loss.item()

            detail_optimizer.zero_grad()
            diffusion_optimizer.zero_grad()
            loss.backward()
            detail_optimizer.step()
            diffusion_optimizer.step()
            