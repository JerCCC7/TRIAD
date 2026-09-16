# Reproducibility and Release Scope

This package was assembled from `matched_bispectrum_multiplicityloop` and the
standalone ERP visualizer. Publication metadata and the framework figure come
from the supplied ICML manuscript's main file, `example_paper.tex`. The separate
`experiment.tex` is not included by that main file and contains older settings;
it was not used as the release specification.

## Configuration Sources

| Setting | Paper main text | Supplied code / checkpoint |
| --- | --- | --- |
| Payload | 32 bits | 32 bits |
| SH cutoff | 16 | 16 |
| Embedding degrees | 6, 8, 14 | 6, 8, 14 |
| Evaluation ERP | 512 x 1024 | test.py uses 256 x 512 |
| Model grid | Not independently specified for training | checkpoint resolution = 256 |
| Training | Adam, 1e-4, 300 epochs | current script: AdamW, 2e-5, cosine decay |
| Objective | MSE weight 1 to 20, BCE weight 10 | current compute_loss: 200*MSE + BCE |
| Initialization | Not fully specified | current script loads an earlier stage checkpoint |

The checkpoint's stored CLI arguments include loss weights that the current
`compute_loss` does not use. They are not a reliable record of the effective loss.
The package exposes both settings explicitly. It does not certify that running
the paper configuration from scratch reproduces the provided checkpoint.

## Model Compatibility

The learned architecture and its parameter names are preserved, including
currently unused registered modules, so that all 342 original state entries
can load strictly. Package imports replace machine-specific `sys.path` entries.
The unused UNetFusion import is removed; the registered UNet is retained.
The original model's encoder internally overrides `middle_layers` to 3, as in
the research implementation; its checkpoint metadata still records 64.

The release exports the current test default `checkpoint_epoch_300.pth` from
the `multiplicityloop3` run, not an arbitrarily chosen best checkpoint. Only
inference tensors and architecture settings are published; optimizer states
and local dataset paths are omitted. `manifest.json` records checksums and size.

## Resolution

Loading does not discard buffers or learnable weights on shape mismatch. The
exported model expects 256 x 512. The manuscript mentions resolution scaling for
baselines but the supplied test entry point does not establish a verified
512 x 1024 TRIAD evaluation protocol. Therefore this release makes no automatic
512-grid checkpoint conversion. A separately validated high-resolution protocol
and exact dataset split would be needed to reproduce all paper tables.

## Rotation Conventions

`legacy` is the function extracted unchanged from the research test. Its grid
uses endpoints and maps `atan2(y,x)/pi` directly to `grid_sample`. As a result,
identity rotation is not an identity pixel mapping (approximately half a
longitude turn plus resampling). It is retained to enable comparison with the
current test, not presented as a corrected geometric operator.

`spherical` uses pixel centres, longitude in [0,2*pi), z-up coordinates, inverse
sampling of active rotations, and circular longitude interpolation. Identity
and exact longitude shifts are tested. The Three.js texture uses the same
coordinate convention. Resampling error is distinct from the theoretical
invariance of the continuous representation.

## Validation Boundaries

The release is checked for strict loading, forward/backward execution,
attack execution, geometry, and demo behavior. Smoke-test results are not paper
benchmark results. No baseline implementations, dataset images, original split
manifests, or claim of exact full-table reproduction is included.

`source_checksums.json` records the source files used during assembly. It is
provenance, not a checksum of the modified release files.
