"""Traditional distortions preserved from the research test implementation."""
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter

def apply_jpeg_compression(img_tensor, quality=60):
    """Apply JPEG compression through a PIL RGB round trip."""
    # img_tensor: [B, C, H, W] in range [-1, 1]
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    compressed_imgs = []
    for b in range(B):
        
        img_np = ((img_tensor[b] + 1) / 2 * 255).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)
        
        
        img_compressed_np = np.array(img_compressed).transpose(2, 0, 1) / 255.0
        img_compressed_tensor = torch.from_numpy(img_compressed_np).float() * 2 - 1
        compressed_imgs.append(img_compressed_tensor)
    
    return torch.stack(compressed_imgs).to(device)


def apply_gaussian_filter(img_tensor, kernel_size=1, sigma=3.0):
    """Apply a per-channel SciPy Gaussian filter."""
    # img_tensor: [B, C, H, W] in range [-1, 1]
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    
    img = (img_tensor + 1) / 2
    
    filtered_imgs = []
    for b in range(B):
        channels = []
        for c in range(C):
            channel = img[b, c].cpu().numpy()
            filtered = gaussian_filter(channel, sigma=sigma)
            channels.append(filtered)
        filtered_imgs.append(np.stack(channels))
    
    filtered = torch.from_numpy(np.stack(filtered_imgs)).float().to(device)
    
    return filtered * 2 - 1


def apply_gaussian_noise(img_tensor, mean=0.0, std=0.05):
    """Add Gaussian noise and clamp to [-1, 1]."""
    noise = torch.randn_like(img_tensor) * std + mean
    return (img_tensor + noise).clamp(-1, 1)


def apply_median_filter(img_tensor, kernel_size=3):
    """Apply a per-channel median filter."""
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    
    img = (img_tensor + 1) / 2
    
    filtered_imgs = []
    for b in range(B):
        channels = []
        for c in range(C):
            channel = img[b, c].cpu().numpy()
            filtered = median_filter(channel, size=kernel_size)
            channels.append(filtered)
        filtered_imgs.append(np.stack(channels))
    
    filtered = torch.from_numpy(np.stack(filtered_imgs)).float().to(device)
    
    return filtered * 2 - 1


def apply_salt_pepper_noise(img_tensor, noise_ratio=0.05):
    """Replace independently sampled channel values with -1 or 1."""
    noisy = img_tensor.clone()
    
    
    salt_mask = torch.rand_like(img_tensor) < (noise_ratio / 2)
    noisy[salt_mask] = 1
    
    
    pepper_mask = torch.rand_like(img_tensor) < (noise_ratio / 2)
    noisy[pepper_mask] = -1
    
    return noisy


def apply_resize(img_tensor, scale=0.5):
    """Downsample and then restore the original image dimensions."""
    B, C, H, W = img_tensor.shape
    
    
    H_small = int(H * scale)
    W_small = int(W * scale)
    img_small = F.interpolate(img_tensor, size=(H_small, W_small), mode='bilinear', align_corners=False)
    
    
    img_resized = F.interpolate(img_small, size=(H, W), mode='bilinear', align_corners=False)
    
    return img_resized


def apply_brightness_adjustment(img_tensor, factor_range=(0.7, 1.3)):
    """Scale image brightness by a randomly sampled factor."""
    factor = torch.empty(1).uniform_(*factor_range).item()
    img = (img_tensor + 1) / 2  
    img_adjusted = (img * factor).clamp(0, 1)
    return img_adjusted * 2 - 1  


def apply_contrast_adjustment(img_tensor, factor_range=(0.7, 1.3)):
    """Scale deviations from the per-channel spatial mean."""
    factor = torch.empty(1).uniform_(*factor_range).item()
    img = (img_tensor + 1) / 2  
    mean = img.mean(dim=[2, 3], keepdim=True)
    img_adjusted = ((img - mean) * factor + mean).clamp(0, 1)
    return img_adjusted * 2 - 1  


def apply_hue_adjustment(img_tensor, hue_range=(-0.1, 0.1)):
    """Apply a randomly sampled hue shift using PIL HSV conversion."""
    hue_factor = torch.empty(1).uniform_(*hue_range).item()
    
    
    img = (img_tensor + 1) / 2
    
    
    
    B, C, H, W = img.shape
    device = img.device
    
    adjusted_imgs = []
    for b in range(B):
        img_np = (img[b] * 255).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        
        img_hsv = img_pil.convert('HSV')
        h, s, v = img_hsv.split()
        h_np = np.array(h, dtype=np.float32)
        h_np = (h_np + hue_factor * 255) % 256
        h = Image.fromarray(h_np.astype(np.uint8))
        img_adjusted = Image.merge('HSV', (h, s, v)).convert('RGB')
        
        
        img_adjusted_np = np.array(img_adjusted).transpose(2, 0, 1) / 255.0
        adjusted_imgs.append(torch.from_numpy(img_adjusted_np).float())
    
    adjusted = torch.stack(adjusted_imgs).to(device)
    
    return adjusted * 2 - 1


def apply_saturation_adjustment(img_tensor, factor_range=(0.7, 1.3)):
    """Interpolate between grayscale and color using a sampled saturation factor."""
    factor = torch.empty(1).uniform_(*factor_range).item()
    
    
    img = (img_tensor + 1) / 2
    
    
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    gray = gray.expand_as(img)
    
    
    img_adjusted = (img - gray) * factor + gray
    img_adjusted = img_adjusted.clamp(0, 1)
    
    
    return img_adjusted * 2 - 1


def apply_random_cropout(img_tensor, crop_ratio=0.2, aspect_ratio_range=(0.5, 2.0), fill_value=-1.0):
    """Replace a random rectangle with a constant value at the requested area ratio."""
    if not (0.0 < crop_ratio < 1.0):
        raise ValueError(f"crop_ratio must be in (0, 1), got: {crop_ratio}")

    attacked = img_tensor.clone()
    B, _, H, W = attacked.shape
    target_area = max(1, int(round(H * W * crop_ratio)))

    for b in range(B):
        aspect_ratio = torch.empty(1).uniform_(*aspect_ratio_range).item()
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))

        crop_h = min(max(crop_h, 1), H)
        crop_w = min(max(crop_w, 1), W)

        
        while crop_h * crop_w > target_area and (crop_h > 1 or crop_w > 1):
            if crop_h >= crop_w and crop_h > 1:
                crop_h -= 1
            elif crop_w > 1:
                crop_w -= 1
            else:
                break

        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()

        replacement = torch.full(
            (1, attacked.shape[1], crop_h, crop_w),
            fill_value,
            device=attacked.device,
            dtype=attacked.dtype
        )

        attacked[b:b+1, :, top:top + crop_h, left:left + crop_w] = replacement

    return attacked


def apply_edge_crop_resize(img_tensor, edge_ratio_h=0.01, edge_ratio_w=0.01):
    """Crop all four edges and resize the remaining image to its original dimensions."""
    if not (0.0 < edge_ratio_h < 1.0 and 0.0 < edge_ratio_w < 1.0):
        raise ValueError(
            f"edge_ratio_h and edge_ratio_w must be in (0, 1), got: {edge_ratio_h}, {edge_ratio_w}"
        )

    _, _, H, W = img_tensor.shape
    remove_h = max(1, int(round(H * edge_ratio_h)))
    remove_w = max(1, int(round(W * edge_ratio_w)))

    if 2 * remove_h >= H or 2 * remove_w >= W:
        raise ValueError(
            f"Crop exceeds image dimensions: remove_h={remove_h}, remove_w={remove_w}, H={H}, W={W}"
        )

    
    cropped = img_tensor[:, :, remove_h:H - remove_h, remove_w:W - remove_w]
    resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    return resized


def apply_edge_crop_blackout(img_tensor, edge_ratio_h=0.01, edge_ratio_w=0.01, fill_value=-1.0):
    """Fill all four edge bands without resizing the image."""
    if not (0.0 < edge_ratio_h < 1.0 and 0.0 < edge_ratio_w < 1.0):
        raise ValueError(
            f"edge_ratio_h and edge_ratio_w must be in (0, 1), got: {edge_ratio_h}, {edge_ratio_w}"
        )

    attacked = img_tensor.clone()
    _, _, H, W = attacked.shape
    remove_h = max(1, int(round(H * edge_ratio_h)))
    remove_w = max(1, int(round(W * edge_ratio_w)))

    attacked[:, :, :remove_h, :] = fill_value
    attacked[:, :, H - remove_h:, :] = fill_value
    attacked[:, :, :, :remove_w] = fill_value
    attacked[:, :, :, W - remove_w:] = fill_value

    return attacked
