# Model Weights

Place the Hugging Face files `model.pt`, `config.json`, and `manifest.json` in
`checkpoints/triad-32bit/`. They are intentionally excluded from Git.

Alternatively, pass an original checkpoint path to `--checkpoint`. The supplied
release is 32-bit, resolution 256 (ERP 256 x 512), epoch 300. Use
`scripts/download_checkpoint.py --repo-id NAMESPACE/TRIAD` after the model repo
is published. No Hugging Face account/repository ID has been assumed.
