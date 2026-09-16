"""Original ERP sampler, retained for research-test compatibility."""
import math
import torch
import torch.nn.functional as F

def get_grid(H, W, device):
    """Generate an endpoint-inclusive spherical Cartesian grid."""
    theta = torch.linspace(0, math.pi, H, device=device)
    phi = torch.linspace(0, 2 * math.pi, W, device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)


def rotate_panorama(img, rot_mat):
    """Apply the historical SO(3) ERP resampling convention."""
    B, C, H, W = img.shape
    device = img.device
    
    grid_xyz = get_grid(H, W, device).unsqueeze(0).expand(B, -1, -1, -1)
    rot_mat_inv = rot_mat.transpose(1, 2)
    grid_xyz_rotated = torch.matmul(grid_xyz, rot_mat_inv)
    
    x, y, z = grid_xyz_rotated.unbind(-1)
    theta = torch.acos(z.clamp(-1 + 1e-6, 1 - 1e-6))
    phi = torch.atan2(y, x)
    
    u = phi / math.pi
    v = 2 * (theta / math.pi) - 1
    
    grid = torch.stack([u, v], dim=-1)
    rotated_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)
    
    return rotated_img

