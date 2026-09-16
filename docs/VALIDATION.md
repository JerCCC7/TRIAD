# Release Validation

Validation date: 2026-09-16. Tests used Python 3.9, PyTorch 2.7.0+cu126,
torchvision 0.22.0, e3nn 0.5.9, and the exported epoch-300 weights.
GPU checks ran on NVIDIA H100 PCIe and NVIDIA A100 80GB PCIe devices.

- All 342 exported state entries were compared to the original checkpoint
  with `torch.equal`; every tensor matched exactly.
- Strict loading succeeded from the source checkout and an independently
  installed wheel outside the original project directory.
- Four CPU tests passed: identity ERP sampling, exact wrapped longitude shift,
  invalid quaternion rejection, and the paper loss-schedule boundaries.
- Two real ERP images completed clean evaluation, two SO(3) rotations each,
  all 13 traditional distortions and both combined distortions.
- One real ERP completed spherical-sampler evaluation at 0, 90 and 180 degrees
  with saved-PNG quantization.
- The paper configuration completed a from-scratch forward/backward optimizer
  step and held-out validation. The fine-tuning configuration completed one
  step from release weights, checkpoint saving, and a resumed second epoch.
- The latency/memory benchmark completed embed, extract and full-forward runs.
  These short runs on shared GPUs are execution checks, not timing claims.
- Chromium/Playwright checks passed at 1440 x 1000 and 390 x 844: nonblank
  textured canvas, pixel changes after dragging, no horizontal overflow,
  successful embedding/export/extraction, distinct repeated-export filenames,
  no compounded rotation, and byte-identical PNG after resetting to identity.
- Original research source checksums remained unchanged.

Commands for CPU checks:

```bash
python -m unittest discover -s tests -v
```

These are small smoke tests, not a replication of the paper's dataset-scale
experiments. Full training, original data splits, baseline comparisons and a
verified 512 x 1024 evaluation protocol have not been rerun or reconstructed.
