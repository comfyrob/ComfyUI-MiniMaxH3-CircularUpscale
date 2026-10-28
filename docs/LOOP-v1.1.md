# LOOP v1.1: preserve the attached workflow

Start with [LOOP-v1.1-Regenerate-3step.json](../workflows/LOOP-v1.1-Regenerate-3step.json).
It is based directly on Rob's attached `LOOP-v1.0-8step_lora.json`, which is
retained under `workflows/sources` for preservation tests.

All retained original nodes keep their positions, sizes, modes and executable
widget settings, except the final save prefix. Prompt/image processing, Gemini
subgraph definitions, seed 111, draft action-speed LoRA, first-pass save and
bypassed effects branches are preserved. Native node titles are restored.
Short explanations are in the existing notes and the new group label.

The existing Model Preview Override stays on the draft branch. There is exactly
one preview node. The refinement model connects directly from MiniMax H3 Loop
to BasicGuider; it has no preview override.

The removed SplitSigmas is replaced by ManualSigmas at the same position. A
new group below the existing canvas holds the clean FL2VA base, one native
Load LoRAs node for Turbo v1.2 + Regenerate rank384, Loop patch, text encoder,
guider and fixed internal audio nodes. The second sampler, circular upscaler,
final decoder and output remain in their original locations.

## Which step count do the sigmas belong to?

The original attachment used `BasicScheduler(simple, 8, denoise=1)` followed by
`SplitSigmas(step=5)`: its refinement was the last three intervals of the
**eight-step draft schedule**.

The new ManualSigmas values form an **independent Regenerate schedule**.
They are not tied to an eight- or twenty-step draft. A list of four sigmas
ending in zero gives three refinement intervals. Changing the draft's step
count does not change a manually specified refinement grid.

Sigma says how noisy the current latent is. For H3's flow convention, refinement
starts from `(1-sigma)*upscaled_latent + sigma*noise`. Lower starting sigma
generally gives more preservation; it is not a guarantee of better detail or
motion. More steps subdivide the refinement path and do not inherently mean
better results for a distilled model.

## Four controlled comparisons

All files preserve the same draft, prompt, seed 111, upscaler, Euler solver,
models and layout. Only the refinement sigmas and final output prefix differ
in executable settings. They do not launch four jobs automatically.

| Workflow | Refinement sigmas (rounded here) | Purpose |
|---|---|---|
| [3step](../workflows/LOOP-v1.1-Regenerate-3step.json) | `0.909375, 0.857143, 0.666667, 0` | Published release default; start here. |
| [3step-start080](../workflows/LOOP-v1.1-Regenerate-3step-start080.json) | `0.8, 0.685714, 0.48, 0` | Lower start with the same three intervals; generated from the published shift-6 function. Experimental comparison. |
| [2step-gentler](../workflows/LOOP-v1.1-Regenerate-2step-gentler.json) | `0.7, 0.494118, 0` | Published lower-noise two-step option. |
| [4step-shift6](../workflows/LOOP-v1.1-Regenerate-4step-shift6.json) | `0.909375, 0.841446, 0.732075, 0.526697, 0` | Original four-step grid from that function, with the same start as the default. |

The JSONs contain higher-precision values. You can instead paste any list into
ManualSigmas in one open graph; unchanged upstream inputs can reuse Comfy's
cached draft/upscale. Keep the prompt-generation outputs fixed while comparing.
For a useful first comparison, try the default versus `3step-start080`.

Source: [NVIDIA's sigma definitions](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/open_h3/sigmas.py)
and [recipe](https://github.com/NVlabs/Sana/blob/sol-engine/models/Open-H3-Regenerate-2K/README.md).
These presets use our working refiner + Turbo setup; Sharp remains omitted
for the basis issue documented in [REGENERATE-LOOP.md](REGENERATE-LOOP.md).

## Should the samplers match?

They do not have to. The sampler is the numerical update rule; sigmas are the
noise levels at which it operates. Your draft uses the REF2VA eight-step Turbo
configuration with `res_multistep`. Regenerate uses different adapters, starts
from an enlarged latent, and its published recipe uses Euler at CFG 1.

Keep the draft's working solver and use Euler for the first refinement tests.
After choosing a sigma grid, changing only the refinement solver is a separate
experiment. Changing both solvers and the sigmas at once makes the cause of an
improvement or regression harder to identify.

## What the audio nodes do

H3's workflow carries video and audio together, even when the refinement's
purpose is visual. The auxiliary audio used by the refiner is separate from
the soundtrack exported into the final video.

| Node(s) | Purpose |
|---|---|
| Separate AV Latent | Splits the video and audio tensors, not the clip's time range. The spatial upscaler receives only the video. |
| Get Audio Latent Size → Simple Math | Converts audio latent length to seconds using `length / 40`. Here 263 positions means 6.575 seconds, including the bookkeeping tail. |
| Empty Audio → VAE Encode Audio | Creates digital silence, then encodes it into H3's representation. Encoded silence is generally not an all-zero latent. |
| Create Solid Mask = 0 → Set Latent Noise Mask | Holds the encoded audio fixed during refinement. A zero mask means preserve; a one mask means denoise. |
| Concat AV Latent | Joins that fixed silent audio with the enlarged video for H3 sampling. |
| MiniMax H3 Loop Decode | Adds wrapped context for decoding and crops to one cycle. Here the visible output is 153 frames and the decoded audio body is 255 positions: both 6.375 seconds. |

Your attachment has audio connected to the final CreateVideo. The revision
therefore takes the **picture from refined Loop Decode** and the **sound from
your existing first-pass Loop Decode**. The bypassed Mirelo branch also receives
that draft audio as its reference. The refiner's internal silence is not exported.
These audio helper nodes do not create the visual loop; the Loop model patch
and Loop Decode handle periodic context.

## Validation

All four graphs pass offline Comfy CLI validation with zero errors and warnings.
The inventory overlay includes the model/image filenames from the attached
workflow; this is not a fresh check of installed files on Modal. No API node
or GPU generation was executed to validate these files.

Preservation tests compare against the exact attached baseline and verify native
titles, original layout/settings, unchanged subgraphs and bypass states, all
unrelated edges, exactly one preview, final audio provenance, and controlled
differences between the four presets. The prior Regenerate workflow was reported
working by Rob; this revised layout has not been GPU-rendered here.

The custom upscaler runtime remains unchanged from v0.1.0. Importing the new
workflow does not require updating the circular upscaler's Python code.
