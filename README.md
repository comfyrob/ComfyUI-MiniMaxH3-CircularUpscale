# MiniMax H3 Circular Latent Upscale

Version 0.1.0. A separate ComfyUI node that adapts LBH's released 3D conv v1
upscaler for latents generated with MiniMax H3 Loop. It does not modify the
installed LBH or loop node packs.

Private development repository:
[`comfyrob/ComfyUI-MiniMaxH3-CircularUpscale`](https://github.com/comfyrob/ComfyUI-MiniMaxH3-CircularUpscale).
For the existing Modal deployment, see [the Modal installation guide](docs/MODAL.md).
This is an independent derivative, with upstream attribution in `NOTICE.md`.

Local experimental branch: two additional **Loop Context Pad / Restore** nodes
can bracket an external H3 video-latent processor. They do not change the
existing circular upscaler. See [the experiment and wiring](docs/LOOP-CONTEXT.md)
for the stock LBH test, measured limitations, and the separate Regenerate path.

## Install and use

For an authenticated GitHub checkout, clone directly into your backend's
`ComfyUI/custom_nodes` directory:

```sh
gh repo clone comfyrob/ComfyUI-MiniMaxH3-CircularUpscale comfyui_minimax_h3_circular_upscale
```

Or install the packaged ZIP:

1. Extract/copy the **`comfyui_minimax_h3_circular_upscale` folder** into the
   `custom_nodes` directory of the ComfyUI installation **that actually runs
   your GPU jobs**. The resulting path must be:

   ```text
   ComfyUI/custom_nodes/comfyui_minimax_h3_circular_upscale/__init__.py
   ```

2. Restart that ComfyUI backend and refresh the browser. No additional runtime
   packages or replacement checkpoint are required in an existing modern
   ComfyUI installation. PyTorch 2.1+ is required for the checkpoint loader.

3. Load the bundled **`MiniMax-H3-2.1-Circular-Latent-Upscale.json`** workflow
   from this folder's `workflows` directory. Select your existing reference image
   if its original filename is unavailable. Your existing checkpoint
   `minimax_h3_latent_upscaler_3d_bf16.safetensors` is supported; keep it in
   `models/latent_upscale_models` or a configured equivalent model directory.

This package is not installed automatically by importing the JSON. A missing
`MiniMaxH3CircularLatentUpscale` node means the backend has not loaded this folder.
A managed service must support installing custom Python nodes to use it.

The workflow is derived from the supplied 2.0, **not diagnostic B**. It retains
both sampling passes and the original reference, prompt, seed, model choices,
FP16 upscaler computation, INT8 video VAE, 2× scale, 158-frame request and 24 fps
export. The output prefix becomes `video/MiniMaxH3_Circular_2Pass`.

## What it fixes

The stock upscaler uses ordinary temporal boundaries even when both sampling
and decoding wrap time. This node:

- Takes only the periodic body: 45 latent positions for the supplied workflow.
- Uses circular **temporal** padding in all 62 time-spanning convolutions,
  including the input/output and residual convolutions. Spatial zero padding
  stays unchanged.
- Runs GroupNorm on that body, without the two duplicated tail positions.
- Uses the original v1 architecture, learned weights and normalization convention.
- Reconstructs the tail by copying the first two upscaled latent positions.
- Returns a synchronously copied CPU latent, rejecting NaN/Inf output instead
  of passing it into decoding. Nonfinite checks do not identify every kind of
  bad-but-finite feature value.

It deliberately has **no temporal-chunking setting**. Full temporal context is
necessary for this implementation. It always releases its own model after
execution and does not change the stock node's cache or model objects.

The log reports `45 periodic positions + 2 tail` and the input/output dimensions
for this workflow. Requested video length remains `17*k+5`; decoded loop length
remains `17*k`. Do not remove frames or add an extra duplicate final frame.

## Validation and limits

The local tests verify cyclic-shift consistency through a full-depth network
with reduced channels, show the stock zero-padded control fails that property,
check every released checkpoint parameter name/shape, exercise tail handling,
spatial padding, input validation, cancellation and a ComfyUI host shim.

The user reported that the original circular-upscaler workflow worked on their
GPU backend. Local automated validation uses CPU tensors and reduced-width
networks; checkpoint compatibility is checked against the published full-size
tensor header. That user report is not a controlled comparison establishing a
universal cure for grid/flash artifacts. The new Pad/Restore experiment and
Regenerate combination have not had a full trained-checkpoint GPU render here.

Changing the trained network's temporal boundary conditions can affect picture
quality. The v1 normalization convention is preserved because the public
training contract does not establish that removing its extra normalization is
correct. The 2.1 workflow also retains the original refinement schedule and VAE
to keep this repair limited to the identified incompatibility.

Supported: separate `[B,24,5*k+2,H,W]` floating video latents, LBH 3D conv v1
weights (`.safetensors` or safe-loaded `.pth`), 1–4× spatial scale. Output spatial
dimensions align to 32 pixels. The default 2× size is exactly 1920×1088.
At 1×, the model is skipped and the loop tail is canonicalized.

Unsupported: 2D/v2/different architectures, joint AV input (use Separate AV
Latent), masks/inpainting, or temporal resizing. Audio bypasses this node via
the existing Separate/Concat AV wiring. MPS may be selected by `auto`, but its
Conv3d behavior and memory use have not been validated here; the supplied
workflow retains `cuda`.

Full-context inference can require substantial VRAM. This node asks ComfyUI to
free memory before loading, but backend convolution workspaces vary. It does
not silently fall back to nonperiodic chunking if memory is insufficient.

## Tests

Tests require `pytest` and PyTorch. From this folder:

```sh
python -m pytest tests -q
```

The header fixture is from the published BF16 checkpoint at Hugging Face
revision `09592c6221ec95cc8e0fae67842e34926c4e668b`. It contains tensor metadata
only, not model weights. See `NOTICE.md` and `LICENSE` for attribution.

To undo: load the original 2.0 workflow. Removing this folder and restarting
ComfyUI removes the added node; the original node packs were not modified.
