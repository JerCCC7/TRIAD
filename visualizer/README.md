# TRIAD Rotation Lab

An interactive interface for embedding a watermark in an equirectangular (ERP)
panorama, inspecting it on a textured 3D sphere, choosing a rotation, and testing
watermark recovery from the exported image. This application is part of the
GitHub source release, including its browser dependencies.

## Requirements

- Install the Python dependencies from the [main README](../README.md#installation).
- Download `model.pt` and `config.json` into `checkpoints/triad-32bit/`, or provide
  an original TRIAD training checkpoint.
- Use a browser with WebGL support. Model inference runs on the Python server;
  the browser handles sphere rendering and rotation interaction.

## Start

From the repository root, with the Python environment activated:

```bash
python visualizer/app.py \
  --checkpoint checkpoints/triad-32bit \
  --data-dir data/test \
  --output-dir outputs/visualizer
```

Open **http://127.0.0.1:7861**. Stop the server with `Ctrl+C`.
The model configuration is read at startup; weights load on the first embedding.
An empty image directory is allowed when uploading an ERP through the browser.

## Workflow

1. **Select the image.** Choose a server-side image and click **Load panorama**,
   or select a local file with the upload control. Input images are converted to
   RGB and resized to the checkpoint's grid.
2. **Embed.** Set **Message seed**, then click **Embed watermark**. This generates
   a binary message and shows clean extraction metrics for the saved watermarked
   image. The sphere displays that image as a texture.
3. **Rotate.** Drag the sphere to select an orientation. The readout shows its
   normalized quaternion `[x, y, z, w]`. The reset button restores identity.
4. **Confirm.** Click **Confirm & extract**. The backend rotates the ERP using the
   chosen quaternion, saves a PNG, and decodes its quantized pixel values.
5. **Inspect.** Compare the three ERP previews, BER, accuracy and bit-error count.
   Download the watermarked PNG, rotated PNG or extraction JSON using the links.

All rotations are absolute relative to the current watermarked image. Confirming
twice does not compound a rotation or overwrite the previous export. Embedding
a new message resets the displayed rotation and metrics.

## Output Files

```text
outputs/visualizer/<session>/
  original.png
  embed_001.png
  embed_001.json
  embed_001_rotation_001.png
  embed_001_rotation_001.json
  embed_001_rotation_002.png
  embed_001_rotation_002.json
```

The embedding report records the seed, model configuration and decoded message.
Each rotation report records `rotation_quaternion_xyzw`, `rotation_matrix`,
`sampler`, `quantization`, `true_bits`, `pred_bits`, `errors`, `num_bits`, `ber`,
and `accuracy`. BER is the number of wrong bits divided by the message length;
accuracy is `1 - BER`. The released model uses 32 bits.

## Command-Line Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--checkpoint` | `checkpoints/triad-32bit` | Exported model directory or original checkpoint |
| `--data-dir` | `data/test` | Directory searched recursively for selectable ERP images |
| `--output-dir` | `outputs/visualizer` | Generated sessions and extraction reports |
| `--device` | `auto` | CUDA when available, otherwise CPU; also accepts `cuda:0` or `cpu` |
| `--host` | `127.0.0.1` | Server listening address |
| `--port` | `7861` | HTTP port |

Default paths are relative to the repository location; explicit relative paths
are resolved from the current working directory. Image dimensions are taken
from the checkpoint, so no manual width or height setting is needed.

## Rotation and Evaluation Conventions

The release demo uses Three.js 0.160.1, a z-up spherical mapping and the
`spherical` rotation sampler. It extracts from the actual saved 8-bit RGB pixel
values. A zero rotation therefore preserves the watermarked PNG. The sphere
and the backend use the same active rotation convention. To evaluate matching
sampler and quantization settings on a dataset:

```bash
python evaluate.py --checkpoint checkpoints/triad-32bit \
  --data-dir data/test --suite rotation \
  --rotation-sampler spherical --png-roundtrip \
  --output-dir outputs/spherical_png
```

This command samples its own messages and rotations; it is not a replay of a
particular interactive session. See the [reproducibility notes](../docs/REPRODUCIBILITY.md)
for differences from the legacy research sampler.

## Remote Access and Troubleshooting

For a remote machine, forward its port through the IDE or establish an SSH tunnel:

```bash
ssh -N -L 7861:127.0.0.1:7861 USER@SERVER
```

Then open http://127.0.0.1:7861 on your own computer. If that local port is in use,
forward a different local port, for example `-L 7862:127.0.0.1:7861`, and open
http://127.0.0.1:7862. This tool has one shared operator state per server process.

| Symptom | Check |
| --- | --- |
| Connection refused | Start the Python server and verify the port or SSH/IDE forwarding |
| Missing model files | Check the model directory contains both `config.json` and `model.pt` |
| Empty image list | Add images under `--data-dir`, or upload an ERP from the browser |
| First embedding is slow | Initial model loading and device initialization happen on that request |
| Blank 3D view | Enable WebGL/hardware acceleration and check browser console errors |
| Unexpected BER difference from CLI | Match the sampler, quantization, input, message and rotation |

## Files Included in GitHub

```text
visualizer/
  app.py                  Python HTTP and model inference server
  README.md               This guide
  static/
    index.html            Application interface
    styles.css            Responsive layout
    app.js                API calls, sphere texture and drag interaction
    vendor/               Three.js, Lucide and their license notices
```

Upload the entire `visualizer/` directory with the repository. Do not upload
generated `outputs/`, private ERP datasets, runtime logs or model weights to
GitHub. Dependencies are bundled locally; no npm build or CDN access is needed.
GitHub stores the application source, but static GitHub Pages hosting alone
cannot run the Python/PyTorch backend. Run the application locally or on a server
with the required Python environment and model files.
