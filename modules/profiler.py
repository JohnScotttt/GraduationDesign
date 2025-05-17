import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
from datetime import datetime
import gc

def create_profiler(output_dir: str):
    """创建 PyTorch Profiler
    
    Args:
        output_dir (str): 输出目录路径
        
    Returns:
        profiler: PyTorch Profiler 对象
    """
    prof_dir = os.path.join(output_dir, 'profiler')
    os.makedirs(prof_dir, exist_ok=True)
    
    return profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

def profile_training(model, train_loader, criterion, optimizer, device, num_batches=10):
    """使用 PyTorch Profiler 分析训练过程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        num_batches: 要分析的批次数量
    """
    try:
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        prof = create_profiler('profiler_output')
        
        model.train()
        with prof:
            for i, (low, gt) in enumerate(train_loader):
                if i >= num_batches:
                    break
                    
                try:
                    # 确保数据在正确的设备上
                    low = low.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)
                    
                    with record_function("forward"):
                        detail_output, diffusion_output = model(low, gt)
                        total_loss, detail_loss, noise_loss, photo_loss, frequency_loss = criterion(
                            (detail_output, diffusion_output), gt)
                    
                    with record_function("backward"):
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                    
                    # 确保每个步骤都被记录
                    prof.step()
                    
                    # 清理不需要的张量
                    del detail_output, diffusion_output, total_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"处理批次 {i} 时发生错误: {str(e)}")
                    continue
        
        # 等待 profiler 完成
        prof.stop()
        
        # 打印性能统计
        print("\n性能分析结果:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10))
        
        # 保存 Chrome 跟踪文件
        prof.export_chrome_trace("profiler_output/trace.json")
        
        print("\n性能分析完成！结果已保存到 profiler_output 目录")
        print("使用以下命令查看详细分析结果：")
        print("tensorboard --logdir=profiler_output")
        
    except Exception as e:
        print(f"性能分析过程中发生错误: {str(e)}")
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 