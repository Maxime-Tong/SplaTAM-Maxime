import torch
import torch.nn.functional as F

def scale_image_tensor(image_tensor, target_scale, mode='bilinear', align_corners=False):
    """
    缩放图像张量到目标比例（支持多通道图像，如RGB或带Alpha的图像）。
    
    参数:
        image_tensor (torch.Tensor): 输入图像张量，形状为 [C, H, W] 或 [B, C, H, W]。
        target_scale (float | tuple): 缩放比例（如 0.5）或目标尺寸 (width, height)。
        mode (str): 插值模式 ('nearest', 'bilinear', 'bicubic')。
        align_corners (bool): 是否对齐像素角点（与PyTorch插值行为一致）。
    
    返回:
        torch.Tensor: 缩放后的图像张量，形状与输入维度一致。
    
    示例:
        >>> # 下采样到 0.5x
        >>> scaled_img = scale_image_tensor(img, 0.5)
        >>> # 上采样到 (512, 256)
        >>> scaled_img = scale_image_tensor(img, (512, 256))
    """
    if not torch.is_tensor(image_tensor):
        raise ValueError("Input must be a torch.Tensor.")
    
    # 统一处理单张图（C,H,W）和批次图（B,C,H,W）
    input_ndim = image_tensor.ndim
    if input_ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    elif input_ndim != 4:
        raise ValueError(f"Expected input shape [C,H,W] or [B,C,H,W], got {image_tensor.shape}")
    
    # 计算目标尺寸
    _, _, h, w = image_tensor.shape
    if isinstance(target_scale, (float, int)):
        target_h, target_w = int(h * target_scale), int(w * target_scale)
    elif isinstance(target_scale, (tuple, list)) and len(target_scale) == 2:
        target_w, target_h = target_scale
    else:
        raise ValueError("target_scale must be float or (width, height) tuple.")
    
    # 执行插值
    scaled_image = F.interpolate(
        image_tensor,
        size=(target_h, target_w),
        mode=mode,
        align_corners=align_corners
    )
    
    # 恢复原始维度
    if input_ndim == 3:
        scaled_image = scaled_image.squeeze(0)  # [1,C,H,W] -> [C,H,W]
    
    return scaled_image

def adjust_learning_rate_global(optimizer, scale_factor):
    """按比例缩放所有参数组的学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale_factor