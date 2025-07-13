import numpy as np
import torch
import open3d as o3d

def build_quaternion(rotation_matrix):
    """
    更稳健的旋转矩阵到四元数转换，处理所有情况
    
    参数:
        rotation_matrix: 形状为(..., 3, 3)的张量
        
    返回:
        形状为(..., 4)的四元数张量，格式为(w, x, y, z)
    """
    # 矩阵的维度检查
    assert rotation_matrix.shape[-2:] == (3, 3), "输入必须是3x3矩阵"
    
    # 计算矩阵对角线元素
    m00 = rotation_matrix[..., 0, 0]
    m11 = rotation_matrix[..., 1, 1]
    m22 = rotation_matrix[..., 2, 2]
    
    # 计算四元数的平方
    qw_sq = 1 + m00 + m11 + m22
    qx_sq = 1 + m00 - m11 - m22
    qy_sq = 1 - m00 + m11 - m22
    qz_sq = 1 - m00 - m11 + m22
    
    # 防止负数开平方
    qw_sq = torch.maximum(qw_sq, torch.zeros_like(qw_sq))
    qx_sq = torch.maximum(qx_sq, torch.zeros_like(qx_sq))
    qy_sq = torch.maximum(qy_sq, torch.zeros_like(qy_sq))
    qz_sq = torch.maximum(qz_sq, torch.zeros_like(qz_sq))
    
    # 计算四元数分量
    qw = 0.5 * torch.sqrt(qw_sq)
    qx = 0.5 * torch.sqrt(qx_sq) * torch.sign(rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2])
    qy = 0.5 * torch.sqrt(qy_sq) * torch.sign(rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0])
    qz = 0.5 * torch.sqrt(qz_sq) * torch.sign(rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1])
    
    # 合并为四元数
    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)
    
    # 归一化
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    return quaternion

def depth_truncation(pt0_3d, pt1_3d, depth1, depth2, percentile):
    depth1_threshold = np.percentile(depth1, percentile)
    depth2_threshold = np.percentile(depth2, percentile)
    valid_mask = (pt0_3d[:, 2] <= depth1_threshold) & (pt1_3d[:, 2] <= depth2_threshold)
    
    pt0_3d_filtered = pt0_3d[valid_mask]
    pt1_3d_filtered = pt1_3d[valid_mask]
    return pt0_3d_filtered, pt1_3d_filtered

def compute_transformation(pt0_3d, pt1_3d, threshold=10.0, max_iterations=200, return_dict=False):
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pt0_3d)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt1_3d)

    trans_init = np.eye(4)
    reg_result = o3d.pipelines.registration.registration_icp(
        pcd0, pcd1, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # R = reg_result.transformation[:3, :3]
    # t = reg_result.transformation[:3, 3:]
    if return_dict:
        return reg_result
    else:
        transformation = reg_result.transformation
        return transformation
    

def back_project(points_2d, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Convert 2D points to pixel coordinates
    u = points_2d[:, 0].astype(int)
    v = points_2d[:, 1].astype(int)
    
    # Get depth values (ensure bounds)
    h, w = depth.shape
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    z = depth[v, u]
    
    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_3d = np.column_stack([x, y, z])
    
    return points_3d



def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
