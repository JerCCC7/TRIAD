#!/usr/bin/env python3
"""Export inference weights and a minimal configuration for Hugging Face."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triad.checkpoint import load_checkpoint, model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if any((output / name).exists() for name in ["model.pt", "config.json", "manifest.json"]):
        raise FileExistsError("Export files already exist")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = model_config(checkpoint.get("model_config", checkpoint.get("args", {})))
    output.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().contiguous() for k, v in checkpoint["model_state_dict"].items()}, output / "model.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Fail export if the published architecture cannot load every weight.
    model, _ = load_checkpoint(output)
    del model
    files = {}
    for name in ["model.pt", "config.json"]:
        file = output / name
        digest = hashlib.sha256()
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        files[name] = {"sha256": digest.hexdigest(), "bytes": file.stat().st_size}
    manifest = dict(format="triad-inference-v1", source_epoch=checkpoint.get("epoch"),
                    source_filename=Path(args.checkpoint).name, optimizer_included=False, files=files)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
