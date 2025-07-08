import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

def expand_mask_to_3x3(mask):
    # (1, 1, H, W)
    mask_float = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)    
    expanded = F.conv2d(mask_float, kernel, padding=1)    
    expanded = expanded.squeeze() > 0
    return expanded

def sample_texture_and_random_mask(image: torch.Tensor, 
    num_samples: int,
    topk: int,
    sobel_kernel_size: int = 3):
    C, H, W = image.shape
    device = image.device
    
    # Step 1: Convert to grayscale if needed
    if image.shape[0] == 3:
        # Use luminance weights for RGB to grayscale conversion
        gray = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])
    else:
        gray = image.squeeze(0)  # Remove channel dim for single-channel input
    
    # Step 2: Compute gradient magnitude using Sobel filters
    # Create Sobel kernels
    kernel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    kernel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Add batch and channel dimensions for conv2d
    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Compute gradients with padding
    Gx = F.conv2d(gray, kernel_x, padding=sobel_kernel_size//2)
    Gy = F.conv2d(gray, kernel_y, padding=sobel_kernel_size//2)
    
    # Calculate gradient magnitude
    G = torch.sqrt(Gx.pow(2) + Gy.pow(2))
    G = G.squeeze()  # Remove batch and channel dims -> (H, W)
    
    # Create edge mask to exclude border pixels affected by padding
    border_size = sobel_kernel_size // 2
    edge_mask = torch.ones((H, W), dtype=bool, device=device)
    if border_size > 0:
        edge_mask[:border_size, :] = False
        edge_mask[-border_size:, :] = False
        edge_mask[:, :border_size] = False
        edge_mask[:, -border_size:] = False
    
    # Flatten and apply edge mask
    G_flat = G.view(-1)
    edge_mask_flat = edge_mask.view(-1)
    valid_indices = torch.where(edge_mask_flat)[0]
    
    # Select topk from valid (non-edge) pixels only
    valid_G = G_flat[valid_indices]
    _, topk_rel_indices = torch.topk(valid_G, min(topk, len(valid_indices)), largest=True)
    topk_indices = valid_indices[topk_rel_indices]
    
    # Random samples also from valid pixels only
    rand_rel_indices = torch.randperm(len(valid_indices), device=device)[:num_samples]
    rand_indices = valid_indices[rand_rel_indices]
    
    mask = torch.zeros((H, W), dtype=bool, device=device)
    mask_flat = mask.view(-1)
    
    mask_flat[topk_indices] = True
    mask_flat[rand_indices] = True
    
    return mask

def sample_random_patches_mask(image_tensor, num_patches, patch_size):
    """
    Vectorized version: Randomly sample patches and return a boolean mask of shape (H, W),
    where True indicates sampled pixels.

    Args:
        image_tensor (torch.Tensor): (C, H, W)
        num_patches (int): number of patches to sample
        patch_size (int): patch size (must be odd recommended)

    Returns:
        torch.BoolTensor: (H, W) mask indicating sampled pixels
    """
    C, H, W = image_tensor.shape
    device = image_tensor.device
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    
    half = patch_size // 2
    min_y, max_y = half, H - half
    min_x, max_x = half, W - half

    if max_y <= min_y or max_x <= min_x:
        raise ValueError("Image too small for given patch size.")

    # Random patch centers
    ys = torch.randint(min_y, max_y, (num_patches,), device=device)
    xs = torch.randint(min_x, max_x, (num_patches,), device=device)

    # Patch offset grid (dy, dx)
    dy = torch.arange(-half, half + 1, device=device)
    dx = torch.arange(-half, half + 1, device=device)
    offset_y, offset_x = torch.meshgrid(dy, dx, indexing='ij')  # shape (patch, patch)

    # shape: (num_patches, patch_size, patch_size)
    patch_y = ys[:, None, None] + offset_y[None, :, :]
    patch_x = xs[:, None, None] + offset_x[None, :, :]

    # Flatten to 1D list of coordinates
    patch_y = patch_y.reshape(-1)
    patch_x = patch_x.reshape(-1)

    # Filter only valid pixels (redundant if sampling is constrained)
    valid = (patch_y >= 0) & (patch_y < H) & (patch_x >= 0) & (patch_x < W)
    patch_y = patch_y[valid]
    patch_x = patch_x[valid]

    # Use advanced indexing to update mask
    mask[patch_y, patch_x] = True

    return mask

def adaptive_random_sampling(
    image: torch.Tensor, 
    num_samples: int, 
    epsilon: float = 0.001,
    sobel_kernel_size: int = 3
) -> torch.Tensor:
    """
    Adaptive random sampling based on texture richness (gradient magnitude).
    Generates a binary mask with exactly num_samples pixels set to 1 (selected).
    
    Args:
        image: Input image tensor (C x H x W) where C=1 (grayscale) or 3 (RGB)
        num_samples: Total number of pixels to sample
        epsilon: Small constant to ensure non-zero sampling probability everywhere
        sobel_kernel_size: Size of Sobel kernel (3 or 5 recommended)
    
    Returns:
        Binary mask tensor (H x W) where 1 indicates sampled pixels
    """
    device = image.device
    
    # Step 1: Convert to grayscale if needed
    if image.shape[0] == 3:
        # Use luminance weights for RGB to grayscale conversion
        gray = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])
    else:
        gray = image.squeeze(0)  # Remove channel dim for single-channel input
    
    # Step 2: Compute gradient magnitude using Sobel filters
    # Create Sobel kernels
    kernel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    kernel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Add batch and channel dimensions for conv2d
    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Compute gradients
    Gx = F.conv2d(gray, kernel_x, padding=sobel_kernel_size//2)
    Gy = F.conv2d(gray, kernel_y, padding=sobel_kernel_size//2)
    
    # Calculate gradient magnitude
    G = torch.sqrt(Gx.pow(2) + Gy.pow(2))
    G = G.squeeze()  # Remove batch and channel dims -> (H, W)
    
    border_size = sobel_kernel_size // 2
    if border_size > 0:
        G[:border_size, :] = epsilon
        G[-border_size:, :] = epsilon
        G[:, :border_size] = epsilon
        G[:, -border_size:] = epsilon
    
    # Step 3: Normalize gradient magnitudes to [0, 1]
    G_min = G.min()
    G_max = G.max()
    G_norm = (G - G_min) / (G_max - G_min + 1e-7)  # Avoid division by zero

    # Step 4: Create sampling probability map
    P = G_norm + epsilon
    P_flat = P.view(-1)  # Flatten to (H*W,)
    P_flat = P_flat / P_flat.sum()  # Normalize to sum to 1
    
    # Step 5: Sample indices according to probabilities
    # Create CDF (cumulative distribution function)
    cdf = torch.cumsum(P_flat, dim=0)
    cdf = cdf / cdf[-1]  # Ensure CDF ends at 1.0
    
    # Generate random numbers and find corresponding indices
    rand_vals = torch.rand(num_samples, device=device)
    sampled_indices = torch.searchsorted(cdf, rand_vals)
    
    # Step 6: Create output mask
    mask_flat = torch.zeros(P_flat.shape, dtype=bool, device=device)
    mask_flat[sampled_indices] = 1
    mask = mask_flat.view(G.shape)  # Reshape to original dimensions
    
    return mask

def generate_random_mask(H, W, num_samples, novelty=None, device='cuda'):
    mask = torch.zeros((H, W), dtype=bool, device=device)
    selected_indices = torch.randperm(H * W, device=device)[:num_samples]    
    mask_flat = mask.view(-1)
    mask_flat[selected_indices] = True
    
    if novelty is not None:
        mask_flat = mask_flat | novelty
        
    return mask

# for tracking
def generate_pixel_mask(image_tensor, tile_size, sparse_fn, **args):
    with torch.no_grad():
        if sparse_fn == 'uniform':
            mask, intensity_map = generate_tile_center_mask(image_tensor.shape, tile_size, **args)
        elif sparse_fn == 'fast':
            mask, intensity_map = generate_tile_fast_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'dog':
            mask, intensity_map = generate_tile_dog_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'harris':
            mask, intensity_map = generate_tile_harris_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'goodFeatures':
            mask, intensity_map = generate_tile_eig_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'orb':
            mask, intensity_map = generate_tile_orb_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'random':
            if image_tensor.shape[0] == 3:
                _, H, W = image_tensor.shape
            else:
                H, W, _ = image_tensor.shape
            num_samples = ((H + tile_size[1] - 1) // tile_size[1]) * ((W + tile_size[0] - 1) // tile_size[0])
            mask = generate_random_mask(H, W, num_samples, **args)

    return mask

def generate_tile_dog_max_response_mask(image_tensor, tile_size, ksize1=5, ksize2=9, device='cuda'):
    """
    Generate a mask with 1 at the point with maximum Difference of Gaussians (DoG) response in each tile,
    and return the intensity mask of the max response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        ksize1 (int): Size of the first Gaussian kernel (smaller sigma).
        ksize2 (int): Size of the second Gaussian kernel (larger sigma).
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max DoG response points, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max DoG response values at mask points, 0 elsewhere.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('float32')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)

    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Apply Gaussian blur with two different kernel sizes
            blur1 = cv2.GaussianBlur(tile, (ksize1, ksize1), 0)
            blur2 = cv2.GaussianBlur(tile, (ksize2, ksize2), 0)
            
            # Compute DoG
            dog = blur1 - blur2
            
            # Find the point with maximum absolute response
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(np.abs(dog))
            
            # Choose the point with maximum absolute response
            if np.abs(min_val) > max_val:
                x, y = min_loc
                intensity = np.abs(min_val)
            else:
                x, y = max_loc
                intensity = max_val
                
            x = left + x
            y = top + y
            if 0 <= x < W and 0 <= y < H:
                mask_np[y, x] = True
                intensity_np[y, x] = intensity
            else:
                # Fallback to center if out of bounds
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # Use the center's DoG response as intensity
                center_in_tile_y = center_y - top
                center_in_tile_x = center_x - left
                if 0 <= center_in_tile_y < tile_h and 0 <= center_in_tile_x < tile_w:
                    intensity_np[center_y, center_x] = np.abs(dog[center_in_tile_y, center_in_tile_x])
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_eig_max_response_mask(image_tensor, tile_size, block_size=2, ksize=3, device='cuda'):
    """
    Divide an image into tiles, compute Shi–Tomasi (min eigenvalue) responses,
    and generate a mask marking the point with the highest min-eigenvalue in each tile.

    Args:
        image (np.ndarray): Input image as a 2D (grayscale) or 3D (color) array.
        tile_size (tuple): (tile_h, tile_w) specifying the size of each tile.
        block_size (int): Neighborhood size for cornerMinEigenVal.
        ksize (int): Aperture size for Sobel operator.

    Returns:
        mask (np.ndarray): Boolean mask of shape (H, W) with True at selected corners.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('float32')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Compute min eigenvalue (Shi–Tomasi) response
            eig = cv2.cornerMinEigenVal(tile, block_size, ksize)
            
            # Clamp negative values (floating-point errors) to zero
            eig = np.clip(eig, a_min=0, a_max=None)

            # Find the location of the maximum response in this tile
            _, max_val, _, max_loc = cv2.minMaxLoc(eig)            
            
            x = left + max_loc[0]
            y = top + max_loc[1]
            mask_np[y, x] = True
            intensity_np[y, x] = max_val
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_harris_max_response_mask(image_tensor, tile_size, block_size=2, ksize=3, k=0.04, device='cuda'):
    """
    Generate a mask with 1 at the Harris corner with maximum response in each tile,
    and return the intensity mask of the max Harris response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        block_size (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter for Sobel operator.
        k (float): Harris detector free parameter.
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max Harris corners, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max Harris response values at mask points, 0 elsewhere.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('float32')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Compute Harris response
            harris = cv2.cornerHarris(tile, block_size, ksize, k)
            
            # Find the point with maximum response
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris)
            
            if max_val > 0:  # Only consider if we found a corner
                x = left + max_loc[0]
                y = top + max_loc[1]
                if 0 <= x < W and 0 <= y < H:
                    mask_np[y, x] = True
                    intensity_np[y, x] = max_val
            else:
                # Fallback to center if no corners detected
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # Use the center's Harris response as intensity (if available)
                center_in_tile_y = center_y - top
                center_in_tile_x = center_x - left
                if 0 <= center_in_tile_y < tile_h and 0 <= center_in_tile_x < tile_w:
                    intensity_np[center_y, center_x] = harris[center_in_tile_y, center_in_tile_x]
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_orb_max_response_mask(image_tensor, tile_size, max_features=1, device='cuda'):
    """
    Generate a mask with 1 at the ORB feature point with maximum response in each tile,
    and return the intensity mask of the max ORB response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        max_features (int): Maximum number of features to detect (we'll use top 1).
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max response ORB features, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max ORB response values at mask points, 0 elsewhere.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('uint8')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(max_features)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Detect ORB features in the tile
            keypoints = orb.detect(tile, None)
            
            if len(keypoints) > 0:
                # Find the keypoint with maximum response
                max_kp = max(keypoints, key=lambda kp: kp.response)
                x = left + int(max_kp.pt[0])
                y = top + int(max_kp.pt[1])
                if 0 <= x < W and 0 <= y < H:
                    mask_np[y, x] = True
                    intensity_np[y, x] = max_kp.response
            else:
                # Fallback to center if no features detected
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # For center fallback, intensity is set to 0 (no feature response)
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_fast_max_response_mask(image_tensor, tile_size, fast_threshold=10, device='cuda'):
    """
    Generate a mask with 1 at the FAST feature point with maximum response in each tile,
    and return the intensity mask of the max FAST response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        fast_threshold (int): Threshold for FAST feature detector.
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max response FAST features, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max FAST response values at mask points, 0 elsewhere.
    """
    # Convert torch tensor to numpy array on CPU
    image_np = image_tensor.cpu().numpy()
    # Convert to HWC format and grayscale for OpenCV
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('uint8')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Initialize FAST detector with nonmax suppression disabled
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, 
                                          nonmaxSuppression=True,
                                          type=cv2.FastFeatureDetector_TYPE_5_8)
    # fast.setNonmaxSuppression(False)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Detect FAST features in the tile with response scores
            keypoints = fast.detect(tile, None)
            
            if len(keypoints) > 0: 
                # Find the keypoint with maximum response
                max_kp = max(keypoints, key=lambda kp: kp.response)
                x = left + int(max_kp.pt[0])
                y = top + int(max_kp.pt[1])
                if 0 <= x < W and 0 <= y < H:
                    mask_np[y, x] = True
                    intensity_np[y, x] = max_kp.response
            else:
                # Fallback to center if no features detected
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # For center fallback, intensity is set to 0 (no feature response)
    
    # Convert back to torch tensor and move to specified device
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_center_mask(image_shape, tile_size, device='cuda'):
    """
    Generate a mask with 1 at the center of each tile over the image,
    and return an intensity mask with 1 at tile centers (since center has no specific intensity).
    
    Args:
        image_shape (tuple): (H, W) size of the image.
        tile_size (tuple): (tile_h, tile_w) size of each tile.
    
    Returns:
        mask (np.ndarray): Boolean/integer mask of shape (H, W),
                           with 1 at tile centers, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with 1 at tile centers, 0 elsewhere.
    """
    if len(image_shape) == 3 and image_shape[0] == 3:
        H, W = image_shape[1:]
    else:
        H, W = image_shape[:2]
    tile_h, tile_w = tile_size

    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)

    # Calculate center positions of tiles
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            center_y = (top + bottom - 1) // 2
            center_x = (left + right - 1) // 2
            mask_np[center_y, center_x] = True
            intensity_np[center_y, center_x] = 1.0  # Center intensity set to 1
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask