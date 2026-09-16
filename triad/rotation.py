"""Active SO(3) rotation of pixel-centred ERP images (z points north)."""
import math

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .legacy_rotation import rotate_panorama as legacy_rotate


def rotate_panorama(image, matrix, mode="spherical"):
    if mode == "legacy":
        return legacy_rotate(image, matrix)
    if mode != "spherical":
        raise ValueError(f"Unknown rotation sampler: {mode}")
    b, c, h, w = image.shape
    theta = (torch.arange(h, device=image.device, dtype=image.dtype) + 0.5) * math.pi / h
    phi = (torch.arange(w, device=image.device, dtype=image.dtype) + 0.5) * 2 * math.pi / w
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    xyz = torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), -1)
    # Row-vector inverse mapping: source = destination @ R for active column R.
    source = torch.einsum("hwj,bjk->bhwk", xyz, matrix.to(image))
    longitude = torch.remainder(torch.atan2(source[..., 1], source[..., 0]), 2 * math.pi)
    latitude = torch.acos(source[..., 2].clamp(-1, 1))
    px = longitude * w / (2 * math.pi) - 0.5
    py = latitude * h / math.pi - 0.5
    # Circular longitude padding makes bilinear interpolation cross the seam.
    padded = torch.cat((image[..., -1:], image, image[..., :1]), dim=-1)
    grid = torch.stack((2 * (px + 1.5) / (w + 2) - 1, 2 * (py + 0.5) / h - 1), -1)
    return F.grid_sample(padded, grid, align_corners=False, padding_mode="border", mode="bilinear")


def quaternion_matrix(values, device="cpu"):
    import numpy as np
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Expected a finite nonzero quaternion [x,y,z,w]")
    return torch.tensor(Rotation.from_quat(q).as_matrix(), dtype=torch.float32, device=device).unsqueeze(0)
