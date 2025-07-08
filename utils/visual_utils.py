import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def visualize_loss_heatmaps(image, gt_image, depth, gt_depth, save_path):
    """
    可视化渲染图像与GT图像之间的loss区域以及深度图的loss区域，并保存到本地
    
    参数:
        image (torch.Tensor): 渲染的RGB图像 [H,W,3] 或 [3,H,W]
        gt_image (torch.Tensor): 原始RGB图像 [H,W,3] 或 [3,H,W]
        depth (torch.Tensor): 渲染的深度图 [H,W] 或 [1,H,W]
        gt_depth (torch.Tensor): 原始深度图 [H,W] 或 [1,H,W]
        save_path (str): 保存路径（包括文件名，如 'output/heatmaps.png'）
    """
    try:
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 确保输入在CPU上并转换为numpy
        device = image.device
        
        # 处理图像数据
        if image.dim() == 3 and image.size(0) == 3:  # [3,H,W] -> [H,W,3]
            image = image.permute(1, 2, 0)
            gt_image = gt_image.permute(1, 2, 0)
        
        image_np = image.detach().cpu().numpy()
        gt_image_np = gt_image.detach().cpu().numpy()
        
        # 计算图像loss (L1 loss)
        image_loss = torch.abs(image - gt_image).mean(dim=-1)  # [H,W]
        image_loss_np = image_loss.detach().cpu().numpy()
        
        # 处理深度数据
        if depth.dim() == 3:  # [1,H,W] -> [H,W]
            depth = depth.squeeze(0)
            gt_depth = gt_depth.squeeze(0)
        
        depth_np = depth.detach().cpu().numpy()
        gt_depth_np = gt_depth.detach().cpu().numpy()
        
        # 计算深度loss (L1 loss)
        depth_loss = torch.abs(depth - gt_depth)
        depth_loss_np = depth_loss.detach().cpu().numpy()
        
        # 创建可视化
        plt.figure(figsize=(18, 12))
        
        # 原始图像和渲染图像
        plt.subplot(2, 3, 1)
        plt.imshow(gt_image_np)
        plt.title('GT Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(image_np)
        plt.title('Rendered Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        sns.heatmap(image_loss_np, cmap='viridis', cbar=True)
        plt.title('Image Loss Heatmap')
        plt.axis('off')
        
        # 原始深度和渲染深度
        plt.subplot(2, 3, 4)
        plt.imshow(gt_depth_np, cmap='plasma')
        plt.title('GT Depth')
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(2, 3, 5)
        plt.imshow(depth_np, cmap='plasma')
        plt.title('Rendered Depth')
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(2, 3, 6)
        sns.heatmap(depth_loss_np, cmap='viridis', cbar=True)
        plt.title('Depth Loss Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"热力图已成功保存到: {save_path}")
    
    except Exception as e:
        print(f"生成热力图时出错: {str(e)}")
        raise

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    H, W = 256, 256
    device = torch.device('cuda')
    
    # 随机生成图像数据 [H,W,3]
    image = torch.rand((H, W, 3), device=device)
    gt_image = torch.rand((H, W, 3), device=device)
    
    # 随机生成深度数据 [H,W]
    depth = torch.rand((H, W), device=device)
    gt_depth = torch.rand((H, W), device=device)
    
    # 可视化并保存
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "loss_heatmaps.png")
    
    visualize_loss_heatmaps(image, gt_image, depth, gt_depth, save_path)