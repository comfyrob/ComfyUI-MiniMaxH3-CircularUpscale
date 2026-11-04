# Native-backed circular upscaler (experimental)

Add **MiniMax H3 Circular Latent Upscale With Model**. The original
**MiniMax H3 Circular Latent Upscale** remains available with the same inputs
and runtime. Existing workflows are not migrated automatically.

## Wire it into the working workflow

1. Add ComfyUI's **Load Latent Upscale Model**. Select the existing
   `minimax_h3_latent_upscaler_3d_bf16.safetensors` checkpoint.
2. Connect its `LATENT_UPSCALE_MODEL` output to the new circular node's `model`.
3. Connect the same video latent currently feeding the original circular node
   to `samples`. A joint H3 AV latent is also accepted; its audio is passed
   through unchanged.
4. Set `width=1920`, `height=1088` for the existing 960×544 → 2× workflow.
5. Connect the new output where the original circular output went. Keep the
   Regenerate Loop model patch, sigma selector, fixed-silence/audio wiring and
   Loop Decode. Add no additional preview. Retain the original node for rollback.

The new node replaces the upscale operation. It does not connect after the
ordinary native upscale node, and it does not replace Regenerate sampling.

ComfyUI must include native H3 upscale support (introduced in commit `35060f0`)
and its matching comfy-kitchen dependency. Precision and device come from the
native loader/model manager; there is no per-node precision override. The
current native loader uses ComfyUI's VAE dtype selection. Match the actual
compute dtype when comparing with the old node's explicit FP16/BF16/FP32 choice.

## What runs

- Use ComfyUI's loaded model, normalization and convolution helpers, and
  weight casting/offloading. Do not reload the checkpoint or own a model cache.
- Remove the two tail positions before any normalization or convolution.
- Compute GroupNorm statistics across the full periodic body. Process layers
  in sections of 16 latent positions, gathering wraparound neighbors at every
  temporal convolution, including the input, output and depthwise layers.
- Rebuild the two-position tail by copying the first two resized positions.
- Preserve latent metadata and the audio tensor. Reject masks rather than
  silently discarding them. Reject nonfinite output and output-dtype overflow.
- Leave model weights, model methods and input tensors unchanged. No global
  patches or ComfyUI source edits. Model residency stays managed by ComfyUI.

Supports the native H3 3D-convolution architecture, not arbitrary models sharing
the `LATENT_UPSCALE_MODEL` socket. Time cannot be resized. Width/height must be
multiples of 32. Conditioning scale uses the average of the actual spatial
ratios, as in the native node; fractional scales rounded by the old multiplier
node may therefore differ. Exact 2× settings match.

This is sectioned execution of the whole-body calculation, not independent
chunk processing. It still stores full hidden activations and is not a
constant-memory algorithm. No GPU speedup or VRAM reduction is claimed yet.

## Modal installation detail

The existing Modal configuration copies an explicit list of runtime files.
Pulling the branch alone is insufficient: include **native_backend.py** and
**native_node.py**, plus **LICENSE-native-GPL-3.0.txt**, in that list. See the
updated copy block in [MODAL.md](MODAL.md). Start/rebuild your usual serve
session yourself; no Modal job is started by installing this source code.

## Validation

The ordinary CPU suite compares the new execution path against the original
full-depth circular network using identical, seeded weights and reduced channel
counts. It covers section boundaries, rotations, constant loops, exact tails,
batching, precision, AV passthrough, invalid inputs, cancellation, and shared
model/input integrity.

`tests/check_native_comfy.py` additionally runs the actual ComfyUI H3 model,
Comfy ops and comfy-kitchen CPU implementation. It checks the original circular
reference, rotation consistency, and unchanged ordinary native output before
and after circular execution. Run in an environment with those dependencies:

```sh
python tests/check_native_comfy.py --comfyui-root /path/to/ComfyUI
```

See [NATIVE-VALIDATION.json](NATIVE-VALIDATION.json) for the recorded run.
These checks use synthetic weights. CUDA kernels, full-width checkpoint output,
actual peak VRAM and the final visual seam still need a GPU run. Compare one
fixed-seed clip using identical dimensions, dtype, Regenerate schedule and VAE.

## Licensing

The new native integration files adapt ComfyUI's execution scheme and are
GPL-3.0-or-later; the original LBH-derived files retain their MIT notice.
See [NOTICE.md](../NOTICE.md) and the included license files.
