#!/usr/bin/env python3
"""Download and verify a published TRIAD model repository."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True, help="Actual Hugging Face namespace/repository")
    p.add_argument("--revision", default="main")
    p.add_argument("--output-dir", default="checkpoints/triad-32bit")
    args = p.parse_args()
    snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=args.output_dir,
                      allow_patterns=["model.pt", "config.json", "manifest.json", "README.md", "LICENSE"])
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ["model.pt", "config.json"]:
        digest = hashlib.sha256()
        with (root / name).open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
    print(f"Verified model files in {root}")


if __name__ == "__main__":
    main()
