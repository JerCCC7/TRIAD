# TRIAD Rotation Lab

From the repository root:

```bash
python visualizer/app.py --checkpoint checkpoints/triad-32bit --data-dir data/test
```

Open http://127.0.0.1:7861. Load or upload an ERP image, embed, drag the sphere,
and confirm. All rotations are absolute relative to the current watermarked
image; confirming twice does not compound a rotation. Re-embedding resets the
rotation and invalidates the previous on-screen result. Every export has a
unique name, with its own quaternion, matrix, true bits and extracted bits.

The release demo uses Three.js 0.160.1, a z-up spherical mapping and the
`spherical` rotation sampler. It extracts from the actual saved 8-bit RGB pixel
values. A zero rotation therefore preserves the watermarked PNG. See
`docs/REPRODUCIBILITY.md` for differences from the legacy research sampler.

The server loads model architecture metadata at startup and weights on first
embedding. One server has one shared operator state. Use an SSH/IDE forwarded
port for remote work. It defaults to loopback and is not an authenticated
public hosting service.
