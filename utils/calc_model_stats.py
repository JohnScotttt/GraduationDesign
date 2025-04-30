import time

import numpy as np
import torch

from models import LowLightEnhancement


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run() -> None:
    # 设置输入尺寸
    input_sizes = [(3, 256, 256), (3, 512, 512), (3, 1024, 1024)]

    # 创建模型
    model = LowLightEnhancement(
        in_channels=3,
        base_channels=64,
        transformer_dim=256,
        patch_size=16,
        num_transformer_layers=4,
        num_heads=8
    )

    # 计算参数量
    param_count = count_parameters(model)
    print(f"模型参数量: {param_count:,} 参数")

    # 计算不同输入尺寸的推理时间
    print("\n不同输入尺寸的推理时间:")
    print("-" * 60)
    print(f"{'输入尺寸':<15} {'推理时间 (ms)':<20}")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    for input_size in input_sizes:
        # 创建随机输入
        x = torch.randn(1, *input_size).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # 计时
        times = []
        iterations = 50

        with torch.no_grad():
            for _ in range(iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()

                _ = model(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转为毫秒

        avg_time = np.mean(times)
        std_time = np.std(times)

        # 输出结果
        size_str = f"{input_size[0]}×{input_size[1]}×{input_size[2]}"
        print(f"{size_str:<15} {avg_time:.2f} ± {std_time:.2f} ms")

    print("-" * 60)

    # 分析每层参数量
    print("\n模型各层参数量:")
    print("-" * 60)
    print(f"{'层名称':<30} {'参数量':<15} {'比例 (%)':<10}")
    print("-" * 60)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count_layer = param.numel()
            percentage = 100.0 * param_count_layer / param_count
            print(f"{name:<30} {param_count_layer:<15,} {percentage:.2f}%")

    print("-" * 60)
    total_size_mb = param_count * 4 / 1024 / 1024  # 假设FP32模型
    print(f"模型总大小 (FP32): {total_size_mb:.2f} MB")
