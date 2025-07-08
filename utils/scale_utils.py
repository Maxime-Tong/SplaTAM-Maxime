import torch
import torch.nn.functional as F
import math

def gaussian_kernel(size: int, sigma: float):
    """生成高斯模糊核"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    
    g = coords**2
    g = (-g / (2 * sigma**2)).exp()
    g /= g.sum()
    
    return g.unsqueeze(0)  # (1, size)

def gaussian_blur(tensor, kernel_size: int, sigma: float):
    """应用高斯模糊"""
    C, H, W = tensor.shape
    
    # 创建水平和垂直核
    kernel = gaussian_kernel(kernel_size, sigma)  # (1, kernel_size)
    kernel_2d = kernel.T @ kernel  # (kernel_size, kernel_size)
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size).to(device=tensor.device)  # (C, 1, kernel_size, kernel_size)
    
    # 添加batch和channel维度
    tensor = tensor.unsqueeze(0)  # (1, C, H, W)
    
    # 应用卷积实现高斯模糊
    padding = kernel_size // 2
    blurred = F.conv2d(
        tensor, kernel_2d, 
        padding=padding, 
        groups=C
    )
    
    return blurred.squeeze(0)  # (C, H, W)

def antialiased_downsample(tensor, scale_factor: float, kernel_size: int = None, sigma: float = None):
    """
    先模糊再下采样，用于去除高频分量
    
    Args:
        tensor (torch.Tensor): (C, H, W) 输入图像
        scale_factor (float): 下采样比例 (0 < scale_factor < 1)
        kernel_size (int): 高斯核大小，默认自动计算
        sigma (float): 高斯核标准差，默认自动计算
        
    Returns:
        torch.Tensor: 下采样后的图像 (C, new_H, new_W)
    """
    # 自动计算高斯模糊参数（基于下采样比例）
    if kernel_size is None:
        # 经验公式：核大小与下采样比例成反比
        kernel_size = max(3, int(round(1.5 / scale_factor)))
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # 确保奇数
    
    if sigma is None:
        # 经验公式：sigma与下采样比例相关
        sigma = 0.5 * (1 / scale_factor) * math.sqrt(2)
    
    # 应用高斯模糊（去除高频分量）
    blurred = gaussian_blur(tensor, kernel_size, sigma)
    
    # 计算目标尺寸
    H, W = tensor.shape[1:]
    new_H = max(1, int(H * scale_factor))
    new_W = max(1, int(W * scale_factor))
    
    # 双三次下采样
    downsampled = F.interpolate(
        blurred.unsqueeze(0), 
        size=(new_H, new_W),
        mode='bicubic',
        align_corners=False
    )
    
    return downsampled.squeeze(0)

def adjust_intrinsics(K, scale_factor):
    """
    Adjust camera intrinsics after downsampling.
    
    Args:
        K (np.ndarray or torch.Tensor): Original intrinsics matrix (3x3).
        scale_factor (int): Downsampling scale factor.
    
    Returns:
        Adjusted intrinsics matrix (3x3).
    """
    K_adjusted = K.clone() if isinstance(K, torch.Tensor) else K.copy()
    K_adjusted[0, 0] /= scale_factor  # fx
    K_adjusted[1, 1] /= scale_factor  # fy
    K_adjusted[0, 2] = (K[0, 2] - 0.5) / scale_factor + 0.5  # cx
    K_adjusted[1, 2] = (K[1, 2] - 0.5) / scale_factor + 0.5  # cy
    return K_adjusted