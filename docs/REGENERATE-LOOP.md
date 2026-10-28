# Regenerate refinement with a periodic timeline

Load [the workflow](../workflows/MiniMax-H3-Regenerate-Loop-Refinement-3Step.json)
in your existing ComfyUI backend, select the reference image, and edit the
shared prompt. Notes and Model Preview Override nodes are included on the canvas.

The data path is:

```text
Original REF2VA draft loop
  → Separate AV Latent
  → existing Circular Latent Upscale
  → Concat with frozen, VAE-encoded silence
  → Regenerate refinement sampler
  → MiniMax H3 Loop Decode
  → silent MP4
```

The refinement model path is separate from the draft:

```text
Clean FL2VA base
  → FL2VA Turbo 4-step v1.2 LoRA, strength 1.0
  → Open Regenerate 2K refiner rank384 LoRA, strength 1.0
  → MiniMax H3 Loop, pad_clips=2, views=1
  → Model Preview Override
  → BasicGuider (CFG 1)
  → SamplerCustomAdvanced (Euler + fresh RandomNoise)
```

The refiner is a LoRA on the H3 diffusion model, not a replacement checkpoint
for the spatial upscaler. Its input is already the enlarged latent. On every
model prediction, the upstream Loop wrapper supplies wrapped end/start context,
keeps the central prediction, and rebuilds its bookkeeping tail. Loop Decode
then decodes wrapped context and returns one cycle. Merely starting from an
already looping latent does not constrain an ordinary refiner's later predictions.

## Models and installed nodes

Additional model files, using the exact filenames in the workflow:

| Folder | File / download |
|---|---|
| `models/diffusion_models` | [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `models/loras` | [minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/30f485ed874aa44c8183348ad361c5eabad318da/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors) |
| `models/loras` | [minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/f5c3b1d5a63e008a691fca1b6096fd57a5307536/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors) |

Use the **ComfyUI-converted** Turbo file above. Its alpha/rank scaling is already
represented in the file; the model-only loader strength stays 1.0. Kijai's
refiner header likewise specifies scale 1.0 baked into its conversion. The two
adapters' matrix shapes were checked against the selected FL2VA base header.

Keep all files from the working draft/circular-upscale workflow, including its
REF2VA model and Turbo adapter, Qwen text encoder, video and audio VAEs, LBH 3D
conv v1 upscaler, and `kijai-taeh3.safetensors` preview decoder.

The graph was checked against Comfy core `426f5ee`, upstream
[Loop `733647e`](https://github.com/pxl-pshr/ComfyUI-MiniMaxH3-Loop/blob/733647e/__init__.py),
the previously used KJ preview schema, and Essentials `9d9f4be` (`SimpleMath+`).
These are the inspected pins in the existing Modal configuration. No additional
custom node class, model wrapper, Comfy core change, or upscaler weight change
is introduced by this branch. The existing v0.1.0 circular-upscaler deployment
can load this JSON once the required models are available.

## Sampling and what this test measures

The draft keeps its original seed, REF2VA references, eight-step schedule,
960×544 canvas and 158-frame request. The circular upscale stays at 2× and FP16.
Refinement outputs 1920×1088; Loop Decode returns 153 frames at 24 fps (6.375 s).
“2K” names the refiner; this test keeps the working canvas for comparison.

The three-step schedule is copied from
[the published Regenerate sigma grid](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/open_h3/sigmas.py):

```text
0.909375, 0.8571428571428571, 0.6666666666666666, 0
```

Refinement uses seed 43 and fresh noise. The initial mixture follows Comfy's H3
flow sampling convention, `(1-sigma)*latent + sigma*noise`. This is substantial
re-noising, so changed fine detail or motion is possible. If needed, the
published gentler two-step alternative is:

```text
0.7, 0.4941176470588235, 0
```

Replace the text in ManualSigmas to try it. Do not obtain refinement sigmas by
splitting the draft schedule. Re-running with only refinement settings changed
lets Comfy reuse the cached draft and upscale while their inputs stay unchanged.

The refinement uses text-only scene conditioning. The visual content comes from
the upscaled latent; draft reference latents are not reintroduced as guide images.
The auxiliary audio is encoded digital silence, sized from the draft's audio
latent length. A zero audio noise mask holds it fixed; the video remains fully
denoised. A zero latent is not substituted for VAE-encoded silence.

## Why Sharp is omitted

NVIDIA's complete release combines the refiner, Turbo and LMS Sharp. Its
standalone implementation evaluates Sharp's curve branches using a timestep
table from **REF2VA**, even though its main transformer is FL2VA. A standard
ComfyUI LoRA loader attaches those branches to the base model's own table.

Direct byte-range reads of the released 1025×8 FP32 tables established:

| Table | SHA-256 |
|---|---|
| REF2VA / the official Sharp basis | `c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7` |
| Selected FL2VA base | `ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e` |

The tables differ. Blindly stacking the Sharp LoRA on this FL2VA model would
therefore not reproduce the published curve calculation. The impact on pictures
has not been measured. This first workflow deliberately tests **refiner + Turbo
with loop-aware sampling**; it is not the complete three-adapter reference recipe.
No approximate basis conversion is silently applied. A faithful Sharp adaptation
would be a separate change after this loop test.

Source: [official adapter manifest](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/blob/main/lora/lora_config.json)
and [adapter implementation](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/open_h3/adapters.py).

## Verification and next render

Offline Comfy CLI validation reports zero errors and warnings. Its catalog uses
published schemas and verified model filenames; it does **not** prove those
files are installed in the target Modal volume.

The 30 repository tests pass, including graph contracts for the independent
model branch, sigma grid, circular upscale, fixed audio, and shared prompt.
The [CPU integration check](../experiments/check_regenerate_loop.py) executes the
pinned upstream Loop wrapper and selected core node code with synthetic model
and VAE stand-ins. It checks all three and two Euler steps, fresh padding noise,
fixed audio, model-clone isolation and decoder cropping. Its
[saved report](../experiments/regenerate-loop-cpu.json) includes source hashes.

```sh
python -m pytest tests -q
python experiments/check_regenerate_loop.py
```

No trained-model GPU render was run: the inspected ComfyUI Modal test slot was
stopped. After rendering, inspect the saved full-quality video in repeated
playback, especially the last-to-first motion. The tiny-VAE preview cannot
establish the final seam or decode quality. Loop-context wiring is verified;
visual seamlessness and refinement quality remain to be established.

This branch starts from v0.1.0 and contains no Context Pad/Restore experiment.
For a local checkout: `git fetch origin` then
`git switch --track origin/codex/regenerate-loop-refinement` (or switch to the
existing local branch). You can also import the workflow JSON directly into
your current installation; the node runtime has not changed.
