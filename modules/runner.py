import dataclasses
import os
import signal
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.cfg_template import params
from modules.dataloader import LowLightDataset
from modules.ddm import EMAHelper
from modules.loss import DetailSimpleLoss, DiffusionLoss
from modules.model import LowLightEnhancement
from utils import load_config, save_config
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

    interrupted_flag = [False]

    def signal_handler(sig, frame):
        if current_rank_var == 0:
            print("\n[!] SIGINT received! Allowing current epoch to finish...")
        interrupted_flag[0] = True

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize DDP parameters as local variables
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank_var = 0
    current_rank_var = 0
    world_size_var = 1
    
    if is_distributed:
        if cfg.settings.device == 'cuda' and torch.cuda.is_available():
            local_rank_var = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank_var)
            dist.init_process_group(backend='nccl', init_method='env://')
            current_rank_var = dist.get_rank()
            world_size_var = dist.get_world_size()
            device = torch.device(f"cuda:{local_rank_var}")
            if current_rank_var == 0:
                print(f"Initializing DDP on rank {current_rank_var} with world size {world_size_var}. Using device: {device}")
        else:
            is_distributed = False # Fallback if CUDA not suitable
            if current_rank_var == 0: # Print only from one process
                 print("Distributed training requested but CUDA is not available/configured or world size <= 1. "
                       "Falling back to single device.")
            # Fallback device setting handled below
    
    if not is_distributed: # Single device setup or fallback
        if cfg.settings.device == 'cuda':
            if not torch.cuda.is_available():
                if current_rank_var == 0: print('CUDA is not available, using CPU instead')
                device = torch.device('cpu')
            else:
                if current_rank_var == 0: print('CUDA is available, using GPU (single)')
                device = torch.device('cuda')
        else:
            if current_rank_var == 0: print('Using CPU')
            device = torch.device('cpu')

    # Directory and experiment name handling
    final_experiment_name = cfg.settings.name
    if current_rank_var == 0:
        os.makedirs(cfg.settings.output_dir, exist_ok=True)
        suffix = 1
        base_name = cfg.settings.name
        final_experiment_name = base_name
        # Ensure unique experiment name by appending suffix if needed
        while os.path.exists(os.path.join(cfg.settings.output_dir, final_experiment_name)):
            final_experiment_name = f"{base_name}_{suffix}"
            suffix += 1
        
        # Create directories for the final experiment name
        exp_dir = os.path.join(cfg.settings.output_dir, final_experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        log_dir = os.path.join(exp_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)

    if is_distributed:
        # Broadcast the final experiment name from rank 0 to all other ranks
        name_container = [final_experiment_name] if current_rank_var == 0 else [None]
        dist.broadcast_object_list(name_container, src=0)
        final_experiment_name = name_container[0]
        dist.barrier() # Ensure rank 0 has created directories

    # All processes define paths using the synchronized final_experiment_name
    exp_dir = os.path.join(cfg.settings.output_dir, final_experiment_name)
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    log_dir = os.path.join(exp_dir, 'log')

    writer = None # Initialize writer for TensorBoard
    if current_rank_var == 0:
        tensorboard_log_path = os.path.join(log_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tensorboard_log_path)

        params_file_path = os.path.join(log_dir, 'params.toml')
        save_config(dataclasses.asdict(cfg), params_file_path)
        
        log_file_path = os.path.join(log_dir, 'training_record.tsv')
        # Write header for the log file. Overwrites if exists from a previous run for this specific experiment.
        with open(log_file_path, 'w') as f:
            f.write("epoch\ttotal_train_loss\tdetail_train_loss\tnoise_train_loss\t"
                    "photo_train_loss\tfrequency_train_loss\tval_psnr\tval_ssim\n")

    # Create datasets using LowLightDataset
    train_dataset = LowLightDataset(tsv_file=cfg.settings.train_tsv_file)
    val_dataset = LowLightDataset(tsv_file=cfg.settings.eval_tsv_file)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size_var,
                                           rank=current_rank_var, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size_var,
                                         rank=current_rank_var, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.settings.batch_size,
            num_workers=cfg.settings.num_workers, sampler=train_sampler, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.settings.batch_size,
            num_workers=cfg.settings.num_workers, sampler=val_sampler, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.settings.batch_size,
            num_workers=cfg.settings.num_workers, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.settings.batch_size,
            num_workers=cfg.settings.num_workers, shuffle=False, pin_memory=True)


    model = LowLightEnhancement(cfg)
    model.to(device)
    if cfg.settings.ckpt != '':
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank_var} if is_distributed and cfg.settings.device == 'cuda' else device
        model.load_state_dict(torch.load(cfg.settings.ckpt, map_location=map_location))

    if is_distributed:
        model = DDP(model, device_ids=[local_rank_var], output_device=local_rank_var, find_unused_parameters=False)
    ema_helper = EMAHelper()
    ema_helper.register(model.module if is_distributed else model)

    # Determine final learning rates (apply scaling if distributed)
    final_detail_lr = cfg.detail.lr
    final_diffusion_lr = cfg.diffusion.lr

    if is_distributed and world_size_var > 1:
        final_detail_lr = cfg.detail.lr * world_size_var
        final_diffusion_lr = cfg.diffusion.lr * world_size_var
        if current_rank_var == 0:
            print(f"[INFO] Distributed training detected (world_size={world_size_var}). Scaling learning rates:")
            print(f"  Detail LR: {cfg.detail.lr} -> {final_detail_lr}")
            print(f"  Diffusion LR: {cfg.diffusion.lr} -> {final_diffusion_lr}")

    # Detail
    # Access original model using .module if distributed
    detail_model_params = (model.module if is_distributed else model).detail_net.parameters()
    if cfg.detail.optimizer == 'Adam':
        detail_optimizer = torch.optim.Adam(
            detail_model_params, lr=final_detail_lr, weight_decay=cfg.detail.weight_decay)
    elif cfg.detail.optimizer == 'SGD':
        detail_optimizer = torch.optim.SGD(
            detail_model_params, lr=final_detail_lr, momentum=cfg.detail.momentum)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.detail.optimizer}")

    if cfg.detail.scheduler == 'StepLR':
        detail_scheduler = torch.optim.lr_scheduler.StepLR(
            detail_optimizer, step_size=cfg.detail.step_size, gamma=cfg.detail.factor)
    elif cfg.detail.scheduler == 'ReduceLROnPlateau':
        detail_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            detail_optimizer, mode='max', factor=cfg.detail.factor,
            patience=cfg.detail.patience, min_lr=cfg.detail.min_lr)
    else:
        raise ValueError(f"Invalid scheduler: {cfg.detail.scheduler}")

    detail_criterion = DetailSimpleLoss().to(device)

    # Diffusion
    # Access original model using .module if distributed
    diffusion_model_params = (model.module if is_distributed else model).diffusion_net.parameters()
    if cfg.diffusion.optimizer == 'Adam':
        diffusion_optimizer = torch.optim.Adam(
            diffusion_model_params, lr=final_diffusion_lr, weight_decay=cfg.diffusion.weight_decay,
            betas=(0.9, 0.999), amsgrad=cfg.diffusion.amsgrad, eps=cfg.diffusion.eps)
    elif cfg.diffusion.optimizer == 'SGD':
        diffusion_optimizer = torch.optim.SGD(
            diffusion_model_params, lr=final_diffusion_lr, momentum=cfg.diffusion.momentum)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.diffusion.optimizer}")

    if cfg.diffusion.scheduler == 'StepLR':
        diffusion_scheduler = torch.optim.lr_scheduler.StepLR(
            diffusion_optimizer, step_size=cfg.diffusion.step_size, gamma=cfg.diffusion.factor)
    elif cfg.diffusion.scheduler == 'ReduceLROnPlateau':
        diffusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            diffusion_optimizer, mode='max', factor=cfg.diffusion.factor,
            patience=cfg.diffusion.patience, min_lr=cfg.diffusion.min_lr)
    else:
        raise ValueError(f"Invalid scheduler: {cfg.diffusion.scheduler}")

    diffusion_criterion = DiffusionLoss().to(device)

    if cfg.settings.early_stop > 0:
        early_stopping = EarlyStopping(patience=cfg.settings.early_stop)

    best_score = 0.0

    epoch_iterator = range(cfg.settings.epochs)
    if current_rank_var == 0:
        epoch_iterator = tqdm(epoch_iterator, desc=f"Training ({final_experiment_name})", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_iterator:
        cudnn.benchmark = True
        # train
        model.train()
        if is_distributed: 
            train_loader.sampler.set_epoch(epoch) 

        epoch_total_loss_sum = 0.0
        epoch_detail_loss_sum = 0.0
        epoch_noise_loss_sum = 0.0
        epoch_photo_loss_sum = 0.0
        epoch_frequency_loss_sum = 0.0
        
        # Create tqdm for training batches only on rank 0
        train_batch_iterator = train_loader
        if current_rank_var == 0:
            train_batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.settings.epochs} Training",
                                        unit="batch", leave=False, dynamic_ncols=True)

        for batch_idx, (low, gt) in enumerate(train_batch_iterator):
            low = low.to(device)
            gt = gt.to(device)

            detail_output, diffusion_output = model(low, gt)
            detail_loss = detail_criterion(detail_output, gt)
            noise_loss, photo_loss, frequency_loss = diffusion_criterion(diffusion_output, gt)
            loss = detail_loss + noise_loss + photo_loss + frequency_loss

            epoch_total_loss_sum += loss.item()
            epoch_detail_loss_sum += detail_loss.item()
            epoch_noise_loss_sum += noise_loss.item()
            epoch_photo_loss_sum += photo_loss.item()
            epoch_frequency_loss_sum += frequency_loss.item()

            detail_optimizer.zero_grad()
            diffusion_optimizer.zero_grad()
            loss.backward()
            detail_optimizer.step()
            diffusion_optimizer.step()
            ema_helper.update(model.module if is_distributed else model)

            if current_rank_var == 0 and isinstance(train_batch_iterator, tqdm):
                train_batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
        if current_rank_var == 0 and isinstance(train_batch_iterator, tqdm):
            train_batch_iterator.close() # Ensure inner pbar is closed

        # Aggregate training losses if distributed
        # These are sums of batch losses from each rank
        # After all_reduce, rank 0 will have the global sum of batch losses
        if is_distributed:
            for loss_name_sum_str in ['epoch_total_loss_sum', 'epoch_detail_loss_sum',
                                      'epoch_noise_loss_sum', 'epoch_photo_loss_sum', 'epoch_frequency_loss_sum']:
                loss_val = locals()[loss_name_sum_str] 
                loss_tensor = torch.tensor(loss_val).to(device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                if current_rank_var == 0: 
                    locals()[loss_name_sum_str] = loss_tensor.item() 

        avg_epoch_total_loss = 0.0
        avg_epoch_detail_loss = 0.0
        avg_epoch_noise_loss = 0.0
        avg_epoch_photo_loss = 0.0
        avg_epoch_frequency_loss = 0.0

        if current_rank_var == 0:
            num_train_batches_per_gpu = len(train_loader)
            total_train_batches_globally = num_train_batches_per_gpu * (world_size_var if is_distributed else 1)
            if total_train_batches_globally == 0: total_train_batches_globally = 1
            
            avg_epoch_total_loss = epoch_total_loss_sum / total_train_batches_globally
            avg_epoch_detail_loss = epoch_detail_loss_sum / total_train_batches_globally
            avg_epoch_noise_loss = epoch_noise_loss_sum / total_train_batches_globally
            avg_epoch_photo_loss = epoch_photo_loss_sum / total_train_batches_globally
            avg_epoch_frequency_loss = epoch_frequency_loss_sum / total_train_batches_globally

        # eval
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0

        # Create tqdm for validation batches only on rank 0
        eval_batch_iterator = val_loader
        if current_rank_var == 0:
            eval_batch_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.settings.epochs} Evaluating",
                                       unit="batch", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, (low, gt) in enumerate(eval_batch_iterator):
                low = low.to(device)
                gt = gt.to(device)

                enhance_img = (model.module if is_distributed else model).enhance(low, cfg.settings.weight) 
                total_psnr += calculate_psnr(enhance_img, gt)
                total_ssim += calculate_ssim(enhance_img, gt)
        
        if current_rank_var == 0 and isinstance(eval_batch_iterator, tqdm):
            eval_batch_iterator.close() # Ensure inner pbar is closed

        if is_distributed:
            total_psnr_tensor = torch.tensor(total_psnr).to(device)
            total_ssim_tensor = torch.tensor(total_ssim).to(device)
            dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_ssim_tensor, op=dist.ReduceOp.SUM)
            
            # For metric averaging, consider total samples if data varies per rank
            # Simple average over (len(val_loader) * world_size_var) if batches are consistent
            avg_psnr = total_psnr_tensor.item() / (len(val_loader.dataset) 
                                                   if len(val_loader.dataset) > 0 else (len(val_loader) * world_size_var))
            avg_ssim = total_ssim_tensor.item() / (len(val_loader.dataset) 
                                                   if len(val_loader.dataset) > 0 else (len(val_loader) * world_size_var))

        else:
            avg_psnr = total_psnr / len(val_loader) if len(val_loader) > 0 else 0
            avg_ssim = total_ssim / len(val_loader) if len(val_loader) > 0 else 0

        avg_score = 0.5 * (avg_psnr / 50) + 0.5 * avg_ssim

        if current_rank_var == 0: 
            # Log to file (TSV)
            with open(log_file_path, 'a') as f:
                f.write(f"{epoch+1}\t{avg_epoch_total_loss:.6f}\t{avg_epoch_detail_loss:.6f}\t{avg_epoch_noise_loss:.6f}\t"
                        f"{avg_epoch_photo_loss:.6f}\t{avg_epoch_frequency_loss:.6f}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n")
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/Train/Total', avg_epoch_total_loss, epoch + 1)
                writer.add_scalar('Loss/Train/Detail', avg_epoch_detail_loss, epoch + 1)
                writer.add_scalar('Loss/Train/Noise', avg_epoch_noise_loss, epoch + 1)
                writer.add_scalar('Loss/Train/Photo', avg_epoch_photo_loss, epoch + 1)
                writer.add_scalar('Loss/Train/Frequency', avg_epoch_frequency_loss, epoch + 1)
                writer.add_scalar('Metrics/Val/PSNR', avg_psnr, epoch + 1)
                writer.add_scalar('Metrics/Val/SSIM', avg_ssim, epoch + 1)
                writer.add_scalar('Metrics/Val/Score', avg_score, epoch + 1)

            if avg_score > best_score:
                best_score = avg_score
                torch.save(model.module.state_dict() if is_distributed else model.state_dict(), os.path.join(ckpt_dir, f'best_model.pth'))

            # Update tqdm progress bar postfix with metrics
            if isinstance(epoch_iterator, tqdm):
                epoch_iterator.set_postfix({
                    'TrainLoss': f'{avg_epoch_total_loss:.4f}',
                    'ValPSNR': f'{avg_psnr:.4f}',
                    'ValSSIM': f'{avg_ssim:.4f}'
                })

        if cfg.settings.early_stop > 0:
            stop_decision = False
            if current_rank_var == 0:
                if early_stopping(avg_score):
                    stop_decision = True
            
            if is_distributed:
                stop_tensor = torch.tensor(int(stop_decision)).to(device)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    if current_rank_var == 0: 
                        if isinstance(epoch_iterator, tqdm):
                            epoch_iterator.write("Early stopping triggered.")
                        else:
                            print("Early stopping triggered.")
                    break 
            elif stop_decision: 
                 if current_rank_var == 0: 
                    if isinstance(epoch_iterator, tqdm):
                        epoch_iterator.write("Early stopping triggered.")
                    else:
                        print("Early stopping triggered.")
                 break
        
        # Check for interrupt signal after early stopping and other epoch-end logic
        global_interrupt_decision = False
        if is_distributed:
            # Sync interrupt status across all processes
            current_process_interrupted_tensor = torch.tensor(int(interrupted_flag[0])).to(device)
            # Use MAX so if any process was interrupted, all get the signal
            dist.all_reduce(current_process_interrupted_tensor, op=dist.ReduceOp.MAX) 
            if current_process_interrupted_tensor.item() == 1:
                global_interrupt_decision = True
        else: # Not distributed
            if interrupted_flag[0]:
                global_interrupt_decision = True
        
        if global_interrupt_decision:
            if current_rank_var == 0:
                message = "\n[!] Interrupt confirmed. Training will stop after this epoch. Saving last model..."
                if isinstance(epoch_iterator, tqdm):
                    epoch_iterator.write(message) # tqdm-friendly way to print
                else:
                    print(message)
            break # Break from the epoch loop for all processes

        if cfg.detail.scheduler == 'StepLR':
            detail_scheduler.step()
        elif cfg.detail.scheduler == 'ReduceLROnPlateau':
            detail_scheduler.step(avg_score)
        
        if cfg.diffusion.scheduler == 'StepLR':
            diffusion_scheduler.step()
        elif cfg.diffusion.scheduler == 'ReduceLROnPlateau':
            diffusion_scheduler.step(avg_score)

    if current_rank_var == 0 and isinstance(epoch_iterator, tqdm):
        epoch_iterator.close()

    if current_rank_var == 0: 
        if interrupted_flag[0]:
            print("Training was interrupted. Saving final model state as last_model.pth")
        else:
            print("Training finished. Saving final model state as last_model.pth")
        torch.save(model.module.state_dict() if is_distributed else model.state_dict(), os.path.join(ckpt_dir, f'last_model.pth'))
        
        if writer: # Close TensorBoard writer
            writer.close()

    if is_distributed:
        dist.destroy_process_group()