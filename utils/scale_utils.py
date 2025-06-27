import torch

def downsample_center(tensor, scale_factor):
    """
    Downsample by taking the center of each tile.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W).
        scale_factor (int): Downsampling scale factor (e.g., 2 for half resolution).
    
    Returns:
        torch.Tensor: Downsampled tensor of shape (C, H//scale_factor, W//scale_factor).
    """
    C, H, W = tensor.shape
    assert H % scale_factor == 0 and W % scale_factor == 0, "Height and width must be divisible by scale_factor."
    
    # Reshape to extract tiles
    tensor = tensor.reshape(C, H // scale_factor, scale_factor, W // scale_factor, scale_factor)
    
    # Take the center of each tile
    center_idx = scale_factor // 2
    downsampled = tensor[:, :, center_idx, :, center_idx]
    
    return downsampled

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