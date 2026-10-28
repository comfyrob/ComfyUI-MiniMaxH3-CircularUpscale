# Experimental loop context adapter

Local prototype, 2026-10-28. The existing `MiniMaxH3CircularLatentUpscale`
implementation is unchanged. Two additional nodes bracket an external
processor through its LATENT ports. They load no weights and do not patch
ComfyUI, LBH, PyTorch, or a model instance.

## Wiring the stock LBH experiment

```text
Separate AV Latent.video_latent
  -> MiniMax H3 Loop Context Pad (Experimental).video_latent
  -> stock Minimax H3 Latent Upscaler (3D).latent
  -> MiniMax H3 Loop Context Restore (Experimental).video_latent
  -> Concat AV Latent.video_latent

Pad.loop_context -> Restore.loop_context
Separate AV Latent.audio_latent -> Concat AV Latent.audio_latent
```

Keep the original checkpoint, upscale factor, precision, seed, sampler and
decoder settings. Set LBH `enable_temporal_chunking=False`. This experiment
does not correct the independent boundaries created by chunking. Keep
`force_unload=True` and spatial alignment 32 as in the working workflow.

Pad discards the two bookkeeping tail positions, adds periodic video context
on both sides in whole five-latent clips, and adds the next two context
positions. Restore takes the original central body and rebuilds the two
tail positions from its start. It allows the processor to change H/W.
The processor must preserve batch size, channels, temporal length and order.
Restore detects shape changes, but cannot detect reordered frames or a
context wired from a different same-shaped source. Use the matching pair.

These nodes accept separate 24-channel H3 video latents of length `5*k+2`.
They reject masked inputs, joint AV latents, empty tensors and NaN/Inf.
They return new sample tensors and preserve the other dictionary entries.
The extra context object contains only lengths and batch size, no model or
tensor references. Audio bypasses this video-only experiment unchanged.

Start with `pad_clips=3` to measure a small context. On the existing 47-position
latent, this processes 77 positions. `pad_clips=15` processes 197 positions
and covers the v1 convolutional radius of 74 positions, but is still not
equivalent to our circular upscaler. Memory requirements rise with context.
Do not put this pair around our circular upscaler: it already handles every
temporal convolution internally and expects the actual periodic body.

## What the local experiment establishes

`experiments/context_boundary.py` compares identical synthetic weights in
full-depth, reduced-width copies of the LBH-v1 structure. It rotates the
input by one whole five-position clip, processes it, unrotates it, and measures
whether choosing a different start of the loop changes the result.

External padding reduces the error but does not eliminate it. GroupNorm
uses statistics across the entire processed temporal extent; errors at the
outer padded edges can therefore affect the central output, even beyond the
convolutional receptive field. Merely increasing the input padding does not
make the stock architecture circular. Rebuilding a duplicated tail also does
not prove the decoded video has a clean seam.

The unit tests include a local convolution where this adapter matches circular
processing, and a convolution + GroupNorm counterexample where it does not.
This is a deliberate limit of the approach, not a missing test tolerance.
The separate full-depth experiment records its measurements in
`experiments/context-results.json`. These are CPU structural tests, not
trained-checkpoint inference, pixel-quality measurements, or GPU renders.

Results from the recorded FP32 CPU run (45 body positions, synthetic weights):

| Processing | Positions processed | Maximum rotation-consistency error |
|---|---:|---:|
| Stock-style, no additional context | 47 | 1.06274 |
| External context, 3 clips per side | 77 | 0.09865 |
| External context, 15 clips per side | 197 | 0.04351 |
| Existing circular implementation | 45 | 0.00000253 |

These errors are latent-value differences, not perceptual scores. The separate
`stock_node_smoke.py` also ran the actual LBH `execute` method from revision
`40316cf008b2fd8663263270669eb4da23f89d2c`, with host/output shims and synthetic
32-channel weights. It processed `47 -> 77 -> 47` temporal positions, doubled
the spatial dimensions, preserved input and weights, and matched the
stock-style control network exactly. See `experiments/stock-node-results.json`.

## Open-H3-Regenerate-2K

The official refinement stage consumes an upscaled H3 latent, mixes it with
noise, and integrates predicted velocity with Euler steps before decoding.
There is no intermediate RGB requirement. Its published default sigmas are
`0.909375, 0.8571428571428571, 0.6666666666666666, 0`; the two-step alternative
is `0.7, 0.4941176470588235, 0`, with CFG 1. Its complete adapter recipe includes
the refiner, FL2VA Turbo v1.2 and LMS Sharp; Kijai's rank-384 file contains
the refiner component only.

For the ComfyUI LoRA route, the appropriate first test is:

```text
H3 + refinement LoRAs -> existing MiniMax H3 Loop -> guider/sampler MODEL
our circular upscaler -> Concat AV Latent -> sampler LATENT -> Loop Decode
```

The existing Loop model patch supplies wrapped context at every model
evaluation and handles sampling noise. A one-time Pad/Restore pair does not
provide that guarantee. No additional model patch is known to be required
specifically because the refiner is a LoRA; its complete rendering behavior
with the intended weights, audio conditioning and schedule remains untested.
The reference NVIDIA implementation fixes the audio stream to encoded silence;
the existing ComfyUI workflow is not automatically identical to that recipe.

Do not send this experimental padded video latent directly into the standalone
NVIDIA engine: its public input path validates particular 37-position shapes
and its output path truncates to 121 frames. Our pair requires temporal length
to survive. Neither those conventions nor the normal-versus-loop tail meaning
can be established by channel count alone. Likewise, joint AV sampling requires
appropriately aligned audio and conditioning, not just padded video. This
prototype does not implement that integration or validate the fast VAE.

Sources inspected 2026-10-28:

- [Refinement and decoding implementation](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/open_h3/pipeline.py)
- [NVIDIA input shape validation](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/infer.py)
- [Sigma schedules](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/open_h3/sigmas.py)
- [Official adapter composition](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/tools/merge_adapters.py)
- [Kijai refiner checkpoint (metadata inspected by byte range)](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors)
- [MiniMax H3 Loop model and sampler wrappers](https://github.com/pxl-pshr/ComfyUI-MiniMaxH3-Loop/blob/main/__init__.py)

## Run locally

```sh
python -m pytest tests -q
python experiments/context_boundary.py --output experiments/context-results.json
```

This branch is experimental and has not been submitted upstream. Restart the
ComfyUI backend after installing the changed package to expose both new nodes.
The existing bundled 2.1 workflow still uses the original circular node.
The new `workflows/MiniMax-H3-Loop-Context-Experimental-LBH.json` replaces that
node with this pair around the stock LBH node. It retains the other model,
prompt, sampling and decode settings, uses three context clips and disables
temporal chunking. Its output prefix is `video/MiniMaxH3_Context_Experimental_LBH`.
It passed offline schema validation with zero errors/warnings; it has not been
rendered. Select the same reference image used by your original 2.1 workflow
before running a matched comparison. Rebuild it with the workspace's
`blueprints/build_loop_context_experiment.py`, which uses structured CLI edits.
