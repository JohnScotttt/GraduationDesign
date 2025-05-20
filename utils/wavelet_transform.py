import numpy as np
import pywt
from PIL import Image
import os

def wavelet_transform(image_path, output_dir=None):
    """
    对输入图像进行二维离散小波变换，并保存四个子图
    
    参数:
        image_path (str): 输入图像的路径
        output_dir (str, optional): 输出目录的路径。如果为None，则使用输入图像所在目录
    
    返回:
        tuple: 四个子图的路径 (LL, LH, HL, HH)
    """
    # 读取图像
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 如果是彩色图像，分别对RGB通道进行处理
    if len(img_array.shape) == 3:
        # 对每个通道进行小波变换
        coeffs_r = pywt.dwt2(img_array[:,:,0], 'haar')
        coeffs_g = pywt.dwt2(img_array[:,:,1], 'haar')
        coeffs_b = pywt.dwt2(img_array[:,:,2], 'haar')
        
        # 合并三个通道的结果
        LL = np.stack([coeffs_r[0], coeffs_g[0], coeffs_b[0]], axis=2)
        LH = np.stack([coeffs_r[1][0], coeffs_g[1][0], coeffs_b[1][0]], axis=2)
        HL = np.stack([coeffs_r[1][1], coeffs_g[1][1], coeffs_b[1][1]], axis=2)
        HH = np.stack([coeffs_r[1][2], coeffs_g[1][2], coeffs_b[1][2]], axis=2)
    else:
        # 灰度图像处理
        coeffs = pywt.dwt2(img_array, 'haar')
        LL, (LH, HL, HH) = coeffs
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入图像的文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存四个子图
    output_paths = []
    for subband, name in zip([LL, LH, HL, HH], ['LL', 'LH', 'HL', 'HH']):
        # 归一化到0-255范围
        if len(subband.shape) == 3:  # 彩色图像
            subband_normalized = np.zeros_like(subband, dtype=np.uint8)
            for i in range(3):
                channel = subband[:,:,i]
                channel_normalized = ((channel - channel.min()) * (255.0 / (channel.max() - channel.min()))).astype(np.uint8)
                subband_normalized[:,:,i] = channel_normalized
        else:  # 灰度图像
            subband_normalized = ((subband - subband.min()) * (255.0 / (subband.max() - subband.min()))).astype(np.uint8)
        
        output_path = os.path.join(output_dir, f'{base_name}_{name}.png')
        Image.fromarray(subband_normalized).save(output_path)
        output_paths.append(output_path)
    
    return tuple(output_paths)

if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python wavelet_transform.py <图像路径> [输出目录]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_paths = wavelet_transform(image_path, output_dir)
        print("小波变换完成！输出文件：")
        for path in output_paths:
            print(f"- {path}")
    except Exception as e:
        print(f"处理图像时出错: {str(e)}") 