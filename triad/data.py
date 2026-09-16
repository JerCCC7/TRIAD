"""Deterministic ERP loading without synthetic fallbacks."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PanoramaDataset(Dataset):
    def __init__(self, root, resolution=256, limit=0, manifest=None):
        root = Path(root).expanduser().resolve()
        if manifest:
            self.files = [root / s.strip() for s in Path(manifest).read_text().splitlines() if s.strip()]
        else:
            self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError(f"No images found in {root}")
        self.transform = transforms.Compose([
            transforms.Resize((resolution, 2 * resolution)),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = tensor.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    pixels = ((x + 1) * 127.5).clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)


def png_roundtrip(x):
    return ((x + 1) * 127.5).clamp(0, 255).round() / 127.5 - 1
