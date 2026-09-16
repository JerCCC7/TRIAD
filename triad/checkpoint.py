"""Strict, portable loading of TRIAD weights and model configuration."""
import json
from pathlib import Path

import torch

from .model import MatchedBispectrumSystem

MODEL_DEFAULTS = dict(
    latent_dim=32, resolution=256, lmax=16, middle_layers=64,
    embed_irreps="64x14e + 128x6e + 128x8e",
    hidden_irreps="32x0e + 32x14e + 64x6e + 64x8e + 32x1e + 32x2e + 16x3e + 16x4e + 16x5e + 16x7e",
)


def model_config(metadata):
    return {k: metadata.get(k, v) for k, v in MODEL_DEFAULTS.items()}


def choose_device(value):
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if path.is_dir():
        config = json.loads((path / "config.json").read_text())
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise ValueError("Expected a TRIAD checkpoint or a directory with config.json and model.pt")
        config = checkpoint.get("model_config", checkpoint.get("args", {}))
        state = checkpoint["model_state_dict"]
    config = model_config(config)
    model = MatchedBispectrumSystem(**config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config
