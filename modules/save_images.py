import os
import torch
from torchvision.utils import save_image

def save_enhanced_images(enhanced_images, original_images, gt_images, save_dir, batch_idx):
    """
    保存增强后的图片
    
    Args:
        enhanced_images (torch.Tensor): 增强后的图片
        original_images (torch.Tensor): 原始低光照图片
        gt_images (torch.Tensor): 真实标签图片
        save_dir (str): 保存目录
        batch_idx (int): 当前批次索引
    """
    # 创建保存目录
    os.makedirs(os.path.join(save_dir, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    
    # 保存图片
    for i in range(enhanced_images.size(0)):
        idx = batch_idx * enhanced_images.size(0) + i
        
        # 保存增强后的图片
        save_image(enhanced_images[i], 
                  os.path.join(save_dir, 'enhanced', f'{idx:04d}.png'))
        
        # 保存原始图片
        save_image(original_images[i], 
                  os.path.join(save_dir, 'original', f'{idx:04d}.png'))
        
        # 保存真实标签图片
        save_image(gt_images[i], 
                  os.path.join(save_dir, 'gt', f'{idx:04d}.png')) 