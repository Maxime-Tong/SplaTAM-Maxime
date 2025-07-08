import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)

# def calc_ssim_shuffled_packed(
#     img1: torch.Tensor,
#     img2: torch.Tensor,
#     mask: torch.Tensor = None,
#     window_size: int = 4,
#     size_average: bool = True,
#     stride: int = 4,
#     repeat_time: int = 10
# ) -> torch.Tensor:
#     """
#     Compute SSIM on reshaped, randomly shuffled mask-selected pixels from img1/img2.

#     Args:
#         img1, img2:  (C, H, W) input images
#         mask:        (H, W) boolean or 0/1 mask
#         window_size: SSIM Gaussian window size
#         size_average:whether to return mean SSIM or per-pixel

#     Returns:
#         SSIM over reshaped, masked, shuffled patch
#     """
#     assert img1.shape == img2.shape
#     # assert img1.dim() == 3 and mask.dim() == 2
#     C, H, W = img1.shape
#     device = img1.device
#     dtype = img1.dtype
    
#     # Step 1: Get valid pixel indices from mask
#     if mask is None:
#         N = H * W
#         flat_idx = torch.arange(N, device=device)
#     else:
#         mask = mask.to(dtype=torch.bool, device=device)
#         coords = mask.nonzero(as_tuple=False)  # (N, 2), each row is (y, x)
#         flat_idx = coords[:, 0] * W + coords[:, 1]
#         N = coords.shape[0]
    
#     if N < 4096:
#         print(N)
#         raise ValueError("Too few selected pixels for SSIM computation.")
    
#     patch_height = 64
#     patch_width = 64
#     patch_size = patch_height * patch_width
    
#     index_list = []
#     for _ in range(repeat_time):
#         tmp_index = torch.randperm(N, device=device)[:patch_size]
#         index_list.append(flat_idx[tmp_index])
#     res_index = torch.cat(index_list)            
    
#     # Step 5: Gather pixels and reshape
#     img1_flat = img1.view(C, -1)[:, res_index]  # (C, usable)
#     img2_flat = img2.view(C, -1)[:, res_index]

#     img1_patch = img1_flat.view(1, C, patch_height, patch_width * repeat_time)
#     img2_patch = img2_flat.view(1, C, patch_height, patch_width * repeat_time)
    
#     # Step 6: Create window and compute SSIM
#     loss_fn = SSIM(window_size=window_size, size_average=size_average, stride=stride)
#     return loss_fn(img1_patch, img2_patch)

def calc_ssim_shuffled_packed(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor = None,
    window_size: int = 4,
    stride: int = 4,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute SSIM on pixels selected by mask, after random shuffling and reshaping
    into a square image.

    If mask is None, treat it as full image (all pixels selected).

    Args:
        img1, img2:  (C, H, W) input images
        mask:        (H, W) boolean or 0/1 mask, or None
        window_size: SSIM Gaussian window size
        size_average: whether to return mean SSIM

    Returns:
        SSIM over shuffled and packed region
    """
    assert img1.shape == img2.shape and img1.dim() == 3
    C, H, W = img1.shape
    device = img1.device
    dtype = img1.dtype

    # Step 1: handle None as full mask
    if mask is None:
        mask = torch.ones((H, W), dtype=torch.bool, device=device)

    mask = mask.to(dtype=torch.bool, device=device)
    coords = mask.nonzero(as_tuple=False)  # (N, 2)

    N = coords.shape[0]
    if N < 4:
        raise ValueError("Too few selected pixels for SSIM computation.")

    # Step 2: shuffle the pixel indices
    perm = torch.randperm(N, device=device)
    coords = torch.cat([coords, coords[perm]], dim=0)

    # Step 3: reshape to square
    # patch_height = int(torch.floor(torch.sqrt(torch.tensor(N*2, dtype=torch.float32, device=device))))
    patch_height = 64
    patch_width = N * 2 // patch_height
    usable = patch_height * patch_width
    coords = coords[:usable]

    # Step 4: flatten pixel indices
    flat_idx = coords[:, 0] * W + coords[:, 1]  # (usable,)

    # Step 5: gather pixels and reshape
    img1_flat = img1.view(C, -1)[:, flat_idx]  # (C, usable)
    img2_flat = img2.view(C, -1)[:, flat_idx]

    img1_patch = img1_flat.view(C, patch_height, patch_width).unsqueeze(0)
    img2_patch = img2_flat.view(C, patch_height, patch_width).unsqueeze(0)

    # Step 6: compute SSIM
    loss_fn = SSIM(window_size=window_size, size_average=size_average, stride=stride)
    return loss_fn(img1_patch, img2_patch)

def gradient_loss(pred, gt, mask, eps=1e-6):
    """
    计算在掩码区域及其邻域的梯度损失
    参数:
        pred: [C, H, W] 预测图像
        gt: [C, H, W] 真实图像
        mask: [1, H, W] 二值掩码，指示需要计算损失的区域
        eps: 避免除零的小常数
    返回:
        梯度损失值 (标量)
    """
    # 添加批处理维度
    pred = pred.unsqueeze(0)  # [1, C, H, W]
    gt = gt.unsqueeze(0)      # [1, C, H, W]
    mask = mask.unsqueeze(0).float()  # [1, 1, H, W]

    # 2. 定义Sobel梯度算子
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],  dtype=torch.float32, device=pred.device)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                          dtype=torch.float32, device=pred.device)
    
    # 3. 计算预测图像的梯度
    grad_pred_x = F.conv2d(pred, sobel_x.repeat(pred.shape[1], 1, 1, 1), 
                          padding=1, groups=pred.shape[1])
    grad_pred_y = F.conv2d(pred, sobel_y.repeat(pred.shape[1], 1, 1, 1), 
                          padding=1, groups=pred.shape[1])
    
    # 4. 计算真实图像的梯度
    grad_gt_x = F.conv2d(gt, sobel_x.repeat(gt.shape[1], 1, 1, 1), 
                        padding=1, groups=gt.shape[1])
    grad_gt_y = F.conv2d(gt, sobel_y.repeat(gt.shape[1], 1, 1, 1), 
                        padding=1, groups=gt.shape[1])
    
    # 5. 计算梯度差异 (L1损失)
    grad_diff_x = torch.abs(grad_pred_x - grad_gt_x)
    grad_diff_y = torch.abs(grad_pred_y - grad_gt_y)
    
    # 6. 仅考虑扩展掩码区域
    masked_diff_x = grad_diff_x * mask
    masked_diff_y = grad_diff_y * mask
    
    # 7. 计算有效像素数
    valid_pixels = mask.sum()
    
    # 8. 避免除零错误
    if valid_pixels < eps:
        return torch.tensor(0.0, device=pred.device)
    
    # 9. 计算平均损失
    loss_x = masked_diff_x.sum() / valid_pixels
    loss_y = masked_diff_y.sum() / valid_pixels
    
    return (loss_x + loss_y) / 2

def patch_ncc_loss(pred, gt, mask_patches, patch_size=8, eps=1e-6):
    """
    pred, gt:       [C, H, W] float tensors, range [0,1]
    mask_patches:   [1, H, W] binary mask indicating which pixels belong to selected patches
    patch_size:     int, e.g. 16
    """
    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)
    mask_patches = mask_patches.unsqueeze(0)
    
    B, C, H, W = pred.shape

    # 1) 使用 unfold 把图像切成不重叠的 patch 向量
    #    输出形状 [B, C * patch_size * patch_size, num_patches]
    patches_pred = F.unfold(pred, kernel_size=patch_size, stride=patch_size)
    patches_gt   = F.unfold(gt,   kernel_size=patch_size, stride=patch_size)
    patches_mk   = F.unfold(mask_patches, kernel_size=patch_size, stride=patch_size)

    # 2) 只保留那些 mask_patches 中全 1（或超过一定阈值）的 patch index
    #    这里简单判断：patch 完全被选中（sum == patch_size**2）
    valid = (patches_mk.sum(dim=1) >= patch_size * patch_size * 0.8)  # 阈值可调
    print(valid.sum())
    if valid.sum() == 0:
        return torch.tensor(0., device=pred.device)

    # 3) 计算 per‐patch mean
    #    [B, num_patches] → 只保留有效 patch
    mean_p = patches_pred.mean(dim=1, keepdim=True)  # [B,1,num_patches]
    mean_t = patches_gt.mean(dim=1, keepdim=True)

    # 4) 计算标准差
    var_p = ((patches_pred - mean_p)**2).sum(dim=1, keepdim=True).sqrt()  # [B,1,num_patches]
    var_t = ((patches_gt   - mean_t)**2).sum(dim=1, keepdim=True).sqrt()

    # 5) 计算 NCC = sum((p-μp)*(t-μt)) / (σp*σt + eps) / N
    num = ((patches_pred - mean_p) * (patches_gt - mean_t)).sum(dim=1, keepdim=True)
    denom = var_p * var_t + eps
    ncc = (num / denom) / (patch_size * patch_size)  # [B,1,num_patches]

    # 6) 聚合 loss
    ncc_valid = ncc[..., valid]                    # 只保留有效索引
    loss = (1 - ncc_valid).mean()                  # 平均 over B 和 patches
    return loss