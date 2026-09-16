#!/usr/bin/env python3
"""Benchmark strict-loaded TRIAD at the checkpoint's native resolution."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from triad.checkpoint import choose_device, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/triad-32bit")
    p.add_argument("--device", default="auto")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output", default="outputs/benchmark.json")
    args = p.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1:
        p.error("Invalid run counts")
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    h = config["resolution"]
    x = torch.rand(args.batch_size, 3, h, h*2, device=device) * 2 - 1
    bits = torch.randint(2, (args.batch_size, config["latent_dim"]), device=device).float()
    is_cuda = torch.device(device).type == "cuda"

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)

    result = dict(settings=vars(args), model=config,
                  hardware=torch.cuda.get_device_name(device) if is_cuda else "CPU",
                  torch_version=str(torch.__version__))
    with torch.inference_mode():
        watermarked = model.embed(x, bits)
        for name, fn in [("embed", lambda: model.embed(x, bits)),
                         ("extract", lambda: model.extract(watermarked)),
                         ("forward", lambda: model(x, bits))]:
            for _ in range(args.warmup):
                fn()
            synchronize()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(args.iterations):
                synchronize()
                start = time.perf_counter()
                fn()
                synchronize()
                times.append((time.perf_counter() - start) * 1000)
            result[name] = dict(mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                                p90_ms=float(np.percentile(times, 90)),
                                mean_ms_per_image=float(np.mean(times) / args.batch_size))
            if is_cuda:
                result[name]["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                result[name]["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
