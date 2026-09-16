#!/usr/bin/env python3
"""Evaluate clean, SO(3), traditional, and combined distortion robustness."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torchmetrics.functional import structural_similarity_index_measure
from tqdm import tqdm

from triad import attacks as a
from triad.checkpoint import choose_device, load_checkpoint
from triad.data import PanoramaDataset, png_roundtrip, save_image, seed_everything
from triad.rotation import rotate_panorama


def attack_specs():
    return {
        "jpeg": (a.apply_jpeg_compression, {"quality": 60}),
        "gaussian_blur": (a.apply_gaussian_filter, {"kernel_size": 1, "sigma": 3.0}),
        "gaussian_noise": (a.apply_gaussian_noise, {"mean": 0.0, "std": 0.05}),
        "median_filter": (a.apply_median_filter, {"kernel_size": 3}),
        "salt_pepper": (a.apply_salt_pepper_noise, {"noise_ratio": 0.05}),
        "resize": (a.apply_resize, {"scale": 0.5}),
        "brightness": (a.apply_brightness_adjustment, {"factor_range": [0.7, 1.3]}),
        "contrast": (a.apply_contrast_adjustment, {"factor_range": [0.7, 1.3]}),
        "hue": (a.apply_hue_adjustment, {"hue_range": [-0.1, 0.1]}),
        "saturation": (a.apply_saturation_adjustment, {"factor_range": [0.7, 1.3]}),
        "cropout": (a.apply_random_cropout, {"crop_ratio": 0.05, "fill_value": -1.0}),
        "edge_crop_resize": (a.apply_edge_crop_resize, {"edge_ratio_h": 0.02, "edge_ratio_w": 0.01}),
        "edge_crop_blackout": (a.apply_edge_crop_blackout, {"edge_ratio_h": 0.01, "edge_ratio_w": 0.01, "fill_value": -1.0}),
    }


def independent_seed(seed, image, case):
    digest = hashlib.sha256(f"{seed}:{image}:{case}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/triad-32bit")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--manifest")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--suite", choices=["clean", "rotation", "traditional", "combined", "all"], default="all")
    p.add_argument("--rotation-sampler", choices=["legacy", "spherical"], default="legacy")
    p.add_argument("--num-images", type=int, default=10, help="0 = all")
    p.add_argument("--num-rotations", type=int, default=10)
    p.add_argument("--angles", type=float, nargs="*", help="Fixed angle degrees with uniform random axes")
    p.add_argument("--attacks", nargs="+", choices=list(attack_specs()), help="Subset of traditional attacks")
    p.add_argument("--png-roundtrip", action="store_true", help="Extract from 8-bit saved-image values")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.num_images < 0 or args.num_rotations < 1:
        p.error("num-images must be nonnegative and num-rotations positive")
    if args.angles and not all(np.isfinite(v) for v in args.angles):
        p.error("angles must be finite")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    seed_everything(args.seed)
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    dataset = PanoramaDataset(args.data_dir, config["resolution"], args.num_images, args.manifest)
    output.mkdir(parents=True, exist_ok=True)
    specs = attack_specs()
    rows, traces, quality = [], [], []
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="Panoramas"):
            image_id = str(dataset.files[index].relative_to(Path(args.data_dir).resolve()))
            seed_everything(independent_seed(args.seed, image_id, "watermark"))
            x = dataset[index].unsqueeze(0).to(device)
            bits = torch.randint(2, (1, config["latent_dim"]), device=device).float()
            wm = model.embed(x, bits)
            if args.png_roundtrip:
                wm = png_roundtrip(wm)
            mse = ((wm - x) / 2).square().mean().item()
            quality.append(dict(image=image_id, psnr_db=-10 * np.log10(max(mse, 1e-12)),
                                ssim=structural_similarity_index_measure((wm + 1) / 2, (x + 1) / 2, data_range=1.0).item()))
            if args.save_images:
                save_image(x, output / "images" / f"{index:05d}_original.png")
                save_image(wm, output / "images" / f"{index:05d}_watermarked.png")

            def record(name, y, matrix=None, case=None):
                if args.png_roundtrip:
                    y = png_roundtrip(y)
                pred = model.extract(y) > 0
                errors = (pred != bits.bool()).sum().item()
                rows.append(dict(image=image_id, attack=name, errors=errors, bits=bits.numel(),
                                 ber=errors / bits.numel(), accuracy=1 - errors / bits.numel()))
                traces.append(dict(image=image_id, attack=name, case=case,
                                   true_bits=bits[0].int().cpu().tolist(), pred_bits=pred[0].int().cpu().tolist(),
                                   rotation_matrix=matrix[0].cpu().tolist() if matrix is not None else None))
                if args.save_images:
                    save_image(y, output / "images" / f"{index:05d}_{name}_{case or 0}.png")

            def random_matrix(case, angle=None):
                rng = np.random.default_rng(independent_seed(args.seed, image_id, case))
                if angle is None:
                    rotation = Rotation.random(random_state=rng)
                else:
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    rotation = Rotation.from_rotvec(axis * np.deg2rad(angle))
                return torch.tensor(rotation.as_matrix(), dtype=x.dtype, device=device).unsqueeze(0)

            record("clean", wm)
            if args.suite in ["rotation", "all"]:
                for angle in args.angles or [None]:
                    name = "so3_random" if angle is None else f"so3_angle_{angle:g}"
                    for r in range(args.num_rotations):
                        matrix = random_matrix(f"{name}:{r}", angle)
                        record(name, rotate_panorama(wm, matrix, args.rotation_sampler), matrix, r)
            if args.suite in ["traditional", "all"]:
                for name in args.attacks or specs:
                    seed_everything(independent_seed(args.seed, image_id, name))
                    fn, kwargs = specs[name]
                    record(name, fn(wm, **kwargs))
            if args.suite in ["combined", "all"]:
                for name in ["cropout", "gaussian_blur"]:
                    seed_everything(independent_seed(args.seed, image_id, name + "_so3"))
                    fn, kwargs = specs[name]
                    matrix = random_matrix(name + "_so3")
                    record(name + "_so3", rotate_panorama(fn(wm, **kwargs), matrix, args.rotation_sampler), matrix)
    groups = defaultdict(list)
    for row in rows:
        groups[row["attack"]].append(row)
    summary = {}
    for name, group in groups.items():
        errors, total = sum(r["errors"] for r in group), sum(r["bits"] for r in group)
        summary[name] = dict(ber=errors / total, accuracy=1 - errors / total,
                             errors=errors, bits=total, trials=len(group),
                             accuracy_std=float(np.std([r["accuracy"] for r in group])))
    with (output / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "traces.json").write_text(json.dumps(traces, indent=2))
    result = dict(settings=vars(args), model=config, num_images=len(dataset), summary=summary,
                  attack_parameters={name: params for name, (_, params) in specs.items()},
                  torch_version=str(torch.__version__),
                  quality=dict(mean_psnr_db=float(np.mean([r["psnr_db"] for r in quality])),
                               mean_ssim=float(np.mean([r["ssim"] for r in quality]))), per_image_quality=quality)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"quality": result["quality"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
