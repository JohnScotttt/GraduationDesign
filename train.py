from modules import train
from modules.profiler import profile_training
import torch
from modules.dataloader import get_dataloader
from modules.model import LowLightEnhancement
from modules.loss import LowLightLoss
from modules.cfg_template import params
from utils import load_config

def main():
    # 加载配置
    default_cfg = load_config('cfg/default.toml')
    cfg = params(**default_cfg)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_loader = get_dataloader(
        cfg.settings.eval_tsv_file,
        batch_size=cfg.settings.batch_size,
        num_workers=cfg.settings.num_workers,
        patch_size=cfg.settings.patch_size
    )
    
    # 创建模型
    model = LowLightEnhancement(cfg)
    model.to(device)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.settings.lr, 
        weight_decay=cfg.settings.weight_decay
    )
    criterion = LowLightLoss(cfg.settings.weight).to(device)
    
    # 运行 profiler
    profile_training(model, train_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main()