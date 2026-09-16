#!/usr/bin/env python3
"""Train TRIAD from scratch or initialize from released weights."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from triad import MatchedBispectrumSystem
from triad.checkpoint import choose_device, load_checkpoint
from triad.data import PanoramaDataset, seed_everything


def mse_weight(epoch, config):
    start, end = config["ramp_start"], config["ramp_end"]
    alpha = max(0.0, min(1.0, (epoch - start) / max(1, end - start)))
    return config["lambda_mse_start"] + alpha * (config["lambda_mse_end"] - config["lambda_mse_start"])


def validate(model, loader, device, seed, max_batches):
    errors = bits = samples = 0
    mse_sum = 0.0
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for i, x in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            w = torch.randint(2, (len(x), model.latent_dim), generator=generator, device=device).float()
            result = model(x, w)
            errors += ((result["w_pred"] > 0) != w.bool()).sum().item()
            bits += w.numel()
            samples += len(x)
            mse_sum += ((result["x_wm"] - x) / 2).square().mean((1, 2, 3)).sum().item()
    mse = mse_sum / samples
    return {"ber": errors / bits, "accuracy": 1 - errors / bits,
            "psnr_db_from_mean_mse": -10 * math.log10(max(mse, 1e-12)), "samples": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/paper.json")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--train-manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--output-dir", required=True)
    weights = p.add_mutually_exclusive_group()
    weights.add_argument("--init-checkpoint", help="Start a new optimizer using these weights")
    weights.add_argument("--resume", help="Resume a checkpoint written by this trainer")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch; for smoke tests")
    p.add_argument("--max-val-batches", type=int, default=0)
    args = p.parse_args()
    settings = json.loads(Path(args.config).read_text())
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        settings = resume["settings"]
    t = settings["training"]
    for key in ["epochs", "batch_size"]:
        value = getattr(args, key)
        if value is not None:
            t[key] = value
    if t["epochs"] < 1 or t["batch_size"] < 1 or args.save_every < 1:
        p.error("epochs, batch-size and save-every must be positive")
    if args.max_train_batches < 0 or args.max_val_batches < 0 or args.num_workers < 0:
        p.error("Batch limits and num-workers must be nonnegative")
    device = choose_device(args.device)
    seed_everything(t["seed"])
    if args.resume or args.init_checkpoint:
        model, cfg = load_checkpoint(args.resume or args.init_checkpoint, device)
        settings["model"] = cfg
    else:
        model = MatchedBispectrumSystem(**settings["model"]).to(device)
    resolution = settings["model"]["resolution"]
    train = PanoramaDataset(args.data_dir, resolution, manifest=args.train_manifest)
    val = PanoramaDataset(args.val_dir, resolution, manifest=args.val_manifest)
    if set(p.resolve() for p in train.files) & set(p.resolve() for p in val.files):
        raise ValueError("Training and validation image lists overlap")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory, including when resuming")
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    # Keep the exact split with each local run; dataset images are never bundled.
    (output / "train_files.txt").write_text("\n".join(map(str, train.files)) + "\n")
    (output / "val_files.txt").write_text("\n".join(map(str, val.files)) + "\n")
    shuffle = torch.Generator().manual_seed(t["seed"])
    train_loader = DataLoader(train, batch_size=t["batch_size"], shuffle=True,
                              num_workers=args.num_workers, generator=shuffle)
    val_loader = DataLoader(val, batch_size=t["batch_size"], num_workers=args.num_workers)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[t["optimizer"]]
    optimizer = optimizer_type(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"], eta_min=t["lr"] * 0.1)
                 if t["scheduler"] == "cosine" else None)
    start, best = 0, 1.0
    if resume:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        start, best = resume["epoch"], resume["best_ber"]
        torch.set_rng_state(resume["torch_rng"])
        shuffle.set_state(resume["shuffle_rng"])
        if torch.cuda.is_available() and resume.get("cuda_rng"):
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    if start >= t["epochs"]:
        raise ValueError("Requested total epochs must exceed the resumed epoch")
    for epoch in range(start + 1, t["epochs"] + 1):
        model.train()
        sums = dict(loss=0.0, mse=0.0, bce=0.0)
        samples = 0
        weight = mse_weight(epoch, t)
        for i, x in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{t['epochs']}")):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x = x.to(device)
            w = torch.randint(2, (len(x), model.latent_dim), device=device).float()
            result = model(x, w)
            mse = F.mse_loss(result["x_wm"], x)
            bce = F.binary_cross_entropy_with_logits(result["w_pred"], w)
            loss = weight * mse + t["lambda_bce"] * bce
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in [("loss", loss), ("mse", mse), ("bce", bce)]:
                sums[key] += value.item() * len(x)
            samples += len(x)
        metrics = validate(model, val_loader, device, t["seed"] + 1, args.max_val_batches)
        row = {"epoch": epoch, "lambda_mse": weight, "lr": optimizer.param_groups[0]["lr"],
               "train": {k: v / samples for k, v in sums.items()}, "validation": metrics}
        print(json.dumps(row))
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if scheduler:
            scheduler.step()
        improved = metrics["ber"] < best
        best = min(best, metrics["ber"])
        checkpoint = dict(epoch=epoch, model_state_dict=model.state_dict(),
                          model_config=settings["model"], settings=settings,
                          optimizer_state_dict=optimizer.state_dict(), best_ber=best,
                          scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                          torch_rng=torch.get_rng_state(), shuffle_rng=shuffle.get_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(checkpoint, output / "last.pth")
        if improved:
            torch.save(checkpoint, output / "best.pth")
        if epoch % args.save_every == 0:
            torch.save(checkpoint, output / f"epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    main()
