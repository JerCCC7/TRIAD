"""Traditional distortions preserved from the research test implementation."""
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter

def apply_jpeg_compression(img_tensor, quality=60):
    """应用JPEG压缩攻击"""
    # img_tensor: [B, C, H, W] in range [-1, 1]
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    compressed_imgs = []
    for b in range(B):
        # 转换为PIL图像 [0, 255]
        img_np = ((img_tensor[b] + 1) / 2 * 255).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # JPEG压缩
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)
        
        # 转回tensor [-1, 1]
        img_compressed_np = np.array(img_compressed).transpose(2, 0, 1) / 255.0
        img_compressed_tensor = torch.from_numpy(img_compressed_np).float() * 2 - 1
        compressed_imgs.append(img_compressed_tensor)
    
    return torch.stack(compressed_imgs).to(device)


def apply_gaussian_filter(img_tensor, kernel_size=1, sigma=3.0):
    """应用高斯滤波"""
    # img_tensor: [B, C, H, W] in range [-1, 1]
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # 转换到 [0, 1] 进行处理
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
    # 转回 [-1, 1]
    return filtered * 2 - 1


def apply_gaussian_noise(img_tensor, mean=0.0, std=0.05):
    """应用高斯噪声"""
    noise = torch.randn_like(img_tensor) * std + mean
    return (img_tensor + noise).clamp(-1, 1)


def apply_median_filter(img_tensor, kernel_size=3):
    """应用中值滤波"""
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # 转换到 [0, 1]
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
    # 转回 [-1, 1]
    return filtered * 2 - 1


def apply_salt_pepper_noise(img_tensor, noise_ratio=0.05):
    """应用椒盐噪声"""
    noisy = img_tensor.clone()
    
    # Salt (白点)
    salt_mask = torch.rand_like(img_tensor) < (noise_ratio / 2)
    noisy[salt_mask] = 1
    
    # Pepper (黑点)
    pepper_mask = torch.rand_like(img_tensor) < (noise_ratio / 2)
    noisy[pepper_mask] = -1
    
    return noisy


def apply_resize(img_tensor, scale=0.5):
    """应用缩放攻击"""
    B, C, H, W = img_tensor.shape
    
    # 缩小
    H_small = int(H * scale)
    W_small = int(W * scale)
    img_small = F.interpolate(img_tensor, size=(H_small, W_small), mode='bilinear', align_corners=False)
    
    # 放大回原尺寸
    img_resized = F.interpolate(img_small, size=(H, W), mode='bilinear', align_corners=False)
    
    return img_resized


def apply_brightness_adjustment(img_tensor, factor_range=(0.7, 1.3)):
    """应用亮度调整"""
    factor = torch.empty(1).uniform_(*factor_range).item()
    img = (img_tensor + 1) / 2  # 转到 [0, 1]
    img_adjusted = (img * factor).clamp(0, 1)
    return img_adjusted * 2 - 1  # 转回 [-1, 1]


def apply_contrast_adjustment(img_tensor, factor_range=(0.7, 1.3)):
    """应用对比度调整"""
    factor = torch.empty(1).uniform_(*factor_range).item()
    img = (img_tensor + 1) / 2  # 转到 [0, 1]
    mean = img.mean(dim=[2, 3], keepdim=True)
    img_adjusted = ((img - mean) * factor + mean).clamp(0, 1)
    return img_adjusted * 2 - 1  # 转回 [-1, 1]


def apply_hue_adjustment(img_tensor, hue_range=(-0.1, 0.1)):
    """应用色调调整"""
    hue_factor = torch.empty(1).uniform_(*hue_range).item()
    
    # 转到 [0, 1]
    img = (img_tensor + 1) / 2
    
    # 转换到HSV
    # 使用PIL进行色调调整
    B, C, H, W = img.shape
    device = img.device
    
    adjusted_imgs = []
    for b in range(B):
        img_np = (img[b] * 255).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # 转换到HSV并调整色调
        img_hsv = img_pil.convert('HSV')
        h, s, v = img_hsv.split()
        h_np = np.array(h, dtype=np.float32)
        h_np = (h_np + hue_factor * 255) % 256
        h = Image.fromarray(h_np.astype(np.uint8))
        img_adjusted = Image.merge('HSV', (h, s, v)).convert('RGB')
        
        # 转回tensor
        img_adjusted_np = np.array(img_adjusted).transpose(2, 0, 1) / 255.0
        adjusted_imgs.append(torch.from_numpy(img_adjusted_np).float())
    
    adjusted = torch.stack(adjusted_imgs).to(device)
    # 转回 [-1, 1]
    return adjusted * 2 - 1


def apply_saturation_adjustment(img_tensor, factor_range=(0.7, 1.3)):
    """应用饱和度调整"""
    factor = torch.empty(1).uniform_(*factor_range).item()
    
    # 转到 [0, 1]
    img = (img_tensor + 1) / 2
    
    # 计算灰度图
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    gray = gray.expand_as(img)
    
    # 调整饱和度
    img_adjusted = (img - gray) * factor + gray
    img_adjusted = img_adjusted.clamp(0, 1)
    
    # 转回 [-1, 1]
    return img_adjusted * 2 - 1


def apply_random_cropout(img_tensor, crop_ratio=0.2, aspect_ratio_range=(0.5, 2.0), fill_value=-1.0):
    """应用随机 cropout 攻击，默认替换 20% 区域为黑色。"""
    if not (0.0 < crop_ratio < 1.0):
        raise ValueError(f"crop_ratio 应在 (0, 1) 之间，当前为: {crop_ratio}")

    attacked = img_tensor.clone()
    B, _, H, W = attacked.shape
    target_area = max(1, int(round(H * W * crop_ratio)))

    for b in range(B):
        aspect_ratio = torch.empty(1).uniform_(*aspect_ratio_range).item()
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))

        crop_h = min(max(crop_h, 1), H)
        crop_w = min(max(crop_w, 1), W)

        # 调整矩形大小，尽量让被替换区域接近目标比例。
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
    """应用边缘裁剪攻击：上下左右各裁 1%，再缩放回原尺寸。"""
    if not (0.0 < edge_ratio_h < 1.0 and 0.0 < edge_ratio_w < 1.0):
        raise ValueError(
            f"edge_ratio_h 和 edge_ratio_w 应在 (0, 1) 之间，当前为: {edge_ratio_h}, {edge_ratio_w}"
        )

    _, _, H, W = img_tensor.shape
    remove_h = max(1, int(round(H * edge_ratio_h)))
    remove_w = max(1, int(round(W * edge_ratio_w)))

    if 2 * remove_h >= H or 2 * remove_w >= W:
        raise ValueError(
            f"裁剪尺寸过大: remove_h={remove_h}, remove_w={remove_w}, H={H}, W={W}"
        )

    # 从四个边缘都裁掉，再 resize 回原图尺寸
    cropped = img_tensor[:, :, remove_h:H - remove_h, remove_w:W - remove_w]
    resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    return resized


def apply_edge_crop_blackout(img_tensor, edge_ratio_h=0.01, edge_ratio_w=0.01, fill_value=-1.0):
    """应用边缘裁剪攻击：上下左右边缘置黑，不进行 resize。"""
    if not (0.0 < edge_ratio_h < 1.0 and 0.0 < edge_ratio_w < 1.0):
        raise ValueError(
            f"edge_ratio_h 和 edge_ratio_w 应在 (0, 1) 之间，当前为: {edge_ratio_h}, {edge_ratio_w}"
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

