# LOOP v1.2: organized graph with one sigma selector

Import [LOOP-v1.2-Regenerate-Sigma-Selector.json](../workflows/LOOP-v1.2-Regenerate-Sigma-Selector.json).
In **Sigma presets · select one in Branch**, use the native **Branch** node's
**branch** dropdown. Its four connected Manual Sigmas nodes contain the same
values as the separate v1.1 workflows:

| Choice | Intervals | Initial sigma |
| --- | ---: | ---: |
| 3 steps - default | 3 | 0.909375 |
| 3 steps - start 0.8 | 3 | 0.8 |
| 2 steps - gentler | 2 | 0.7 |
| 4 steps - shift 6 | 4 | 0.909375 |

The default is **3 steps - default**. Branch sends the selected SIGMAS to the
existing Regenerate sampler. The draft retains its independent 8-step
BasicScheduler. The schedule lists are editable in their Manual Sigmas nodes;
input labels identify each choice without changing native node titles.

The graph reads left to right: reference/prompt, draft, circular latent
upscale, Regenerate, final video. Supporting encoders, fixed internal audio,
the first-pass save, prompt helper and bypassed effects have separate groups.
The original prompts, models, seed, settings, subgraph definitions, audio routing
and bypass modes are retained. There is still only the draft Model Preview
Override. The final save prefix is `video/LOOP_v1_2_Regenerate`; ComfyUI adds its
normal counter. Keep the selected preset with your comparison notes.

Branch is a native ComfyUI node, currently marked experimental upstream. A
ComfyUI backend/frontend supporting **BranchNode** is required. No custom-node
runtime files changed. The selector has been checked against its upstream
[backend](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_logic.py)
and [frontend widget implementation](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/extensions/core/customWidgets.ts).
Offline graph validation and preset/preservation checks pass; this revision has
not been GPU-rendered or reloaded in the user's running Modal UI here.

For model setup and audio details, see [v1.1 notes](LOOP-v1.1.md) and
[Regenerate loop refinement](REGENERATE-LOOP.md).
