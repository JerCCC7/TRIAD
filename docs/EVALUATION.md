# Evaluation Protocol

All selected images are evaluated, including traditional attacks and rotations.
The old script's rotation and traditional sections only used the first image;
this release intentionally extends those sections to all selected images.
Each image receives one seeded 32-bit message reused across all attacks.
Random seeds are derived from the global seed, relative image name, and attack
name, so selecting a subset of attacks does not change their random inputs.

| Attack | Parameters |
| --- | --- |
| JPEG | PIL RGB, quality 60; original uint8 truncation preserved |
| Gaussian blur | scipy.ndimage.gaussian_filter, sigma=3 per channel |
| Gaussian noise | mean=0, std=0.05 in [-1,1] tensor space, then clamp |
| Median filter | size=3, per channel |
| Salt and pepper | ratio=0.05, independent channel masks |
| Resize | downsample to 0.5 then upsample, bilinear, align_corners=False |
| Brightness | factor uniformly sampled from [0.7,1.3] |
| Contrast | factor uniformly sampled from [0.7,1.3] |
| Hue | shift uniformly sampled from [-0.1,0.1], PIL HSV |
| Saturation | factor uniformly sampled from [0.7,1.3] |
| Cropout | area ratio 0.05, aspect ratio [0.5,2], fill -1 |
| Edge crop + resize | vertical edge ratio 0.02, horizontal edge ratio 0.01 |
| Edge blackout | both edge ratios 0.01, fill -1, no resize |
| Cropout + SO(3) | cropout above, followed by a random rotation |
| Gaussian blur + SO(3) | blur above, followed by a random rotation |

Gaussian blur's historical `kernel_size=1` argument is unused by the function;
SciPy chooses the support from sigma and its default truncation. The edge-crop
code used a vertical ratio of 0.02 despite its old printed label saying 1%.
The release reports effective parameters rather than those old labels.

SO(3) random rotations use SciPy's uniform rotation sampler. An angle sweep
instead normalizes Gaussian 3-vectors to obtain uniform axes and applies the
specified axis-angle rotation. `--num-rotations` is the number of random rotations
or axes **per image, per angle**. Matrices and all decoded bits are saved.
There is no undocumented "Mixed" attack: the two supported combinations are
reported separately because the paper does not specify one unique Mixed recipe.

Floating-tensor evaluation is the default. The model output is not universally
clamped before extraction. `--png-roundtrip` clamps and rounds both watermarked
and attacked images to 8-bit pixel values before decoding. Saved-image scores
and floating-tensor scores should not be mixed.

Clean PSNR and SSIM compare the original and watermarked RGB values after
mapping [-1,1] to [0,1], with data range 1. PSNR is averaged per image.
Accuracy and BER are aggregated by bit counts. Standard deviation is over
individual image/rotation trials, not a confidence interval. PSNR at exact zero
MSE is capped at 120 dB for JSON portability.
