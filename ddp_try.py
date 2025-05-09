import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from modules import DetailNet
from modules.dataloader import LowLightDataset
from torch.utils.data import DataLoader
from modules import DetailVGGLoss

def main():
    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # 创建模型并包装为DDP
    model = DetailNet().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 数据加载器设置
    train_dataset = LowLightDataset("data/train.tsv")
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    criterion = DetailVGGLoss()
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters())
    
    # 训练循环
    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        for inputs, labels in train_loader:
            inputs = inputs.cuda(local_rank)
            labels = labels.cuda(local_rank)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 可选：仅在rank 0保存模型
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), 'model.pth')
    
    # 清理进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    main()