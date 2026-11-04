# Install in the existing Modal ComfyUI image

This repository is private. The current `comfyui_config.py` clones its
`CUSTOM_NODES` entries anonymously, so putting this private URL in that dictionary
alone will fail. For development, authenticate locally with GitHub and let Modal
copy the checked-out code into its image. No GitHub secret is needed in Modal.

## 1. Check out the node beside the Modal configuration

From the directory containing `comfyui_config.py`, using the GitHub account with
access to this repository:

```sh
mkdir -p vendor
gh repo clone comfyrob/ComfyUI-MiniMaxH3-CircularUpscale vendor/ComfyUI-MiniMaxH3-CircularUpscale
git -C vendor/ComfyUI-MiniMaxH3-CircularUpscale checkout codex/native-circular-upscale
git -C vendor/ComfyUI-MiniMaxH3-CircularUpscale rev-parse HEAD
```

Record the printed commit SHA with your test results. Add
`/vendor/ComfyUI-MiniMaxH3-CircularUpscale/` to the Modal project's `.gitignore`
so its checkout is not accidentally committed inside the deployment repository.

## 2. Add the files at the end of image construction

In `comfyui_config.py`, immediately after the existing
`image = image.add_local_python_source("comfyui_config", copy=True)` line and
before `Shared helpers`, add:

```python
from pathlib import Path

_h3_node_source = (
    Path(__file__).resolve().parent / "vendor" / "ComfyUI-MiniMaxH3-CircularUpscale"
)
_h3_node_target = "/comfyui/custom_nodes/comfyui_minimax_h3_circular_upscale"
for _h3_file in (
    "__init__.py", "core.py", "node.py", "native_backend.py", "native_node.py",
    "LICENSE", "LICENSE-native-GPL-3.0.txt", "NOTICE.md",
):
    image = image.add_local_file(
        str(_h3_node_source / _h3_file),
        f"{_h3_node_target}/{_h3_file}",
        copy=True,
    )
```

This deliberately copies only the runtime files and attribution, not Git
metadata or local credentials. It adds layers after the existing image build.
Editing these local node files causes Modal to include their new contents in
the next build. No additional pip installation is needed.

For an existing clone, fetch the remote branch before checking it out. This
branch retains the original runtime and adds the native-backed node. A Modal
configuration still copying only the original five files must be updated with
the list above before rebuilding. The native node requires the ComfyUI native
H3 upscaler and its matching dependencies; the original node does not.

Keep the existing `ComfyUI-MiniMaxH3-Loop` pack. The original LBH upscaler can
also remain installed; this node has a separate class name.

## 3. Start one test slot and load the workflow

Use the deployment project's existing account/profile checks. Its documented
Modal profile is `rob-73810`; the GitHub identity for that deployment repository
is separate from access to this node repository.

For example, from the deployment folder, start its existing A100 80 GB test slot:

```sh
modal serve comfyui_webui.py::app_a100_80gb_a
```

This command starts billable GPU infrastructure. Use an available slot according
to your normal workflow. The repository publication itself does not start a slot
or deploy any Modal application.

Load `workflows/MiniMax-H3-2.1-Circular-Latent-Upscale.json` from the node checkout
in the ComfyUI browser. Confirm the **MiniMax H3 Circular Latent Upscale** node
loads, then select the original reference image if necessary.

Keep the existing `minimax_h3_latent_upscaler_3d_bf16.safetensors` checkpoint
in the model volume's `latent_upscale_models` directory. The new node does not
download weights. The test preserves both sampling passes and the existing VAE.

The execution log should include:

```text
[H3 Circular Upscale] 45 periodic positions + 2 tail; 960x544 -> 1920x1088
```

No GPU render has been validated for this patch yet. The local tests establish
the periodic boundary behavior; final image quality and the bright-grid failure
still require running the trained model on the backend. See `VALIDATION.json`.

To undo, remove the added image block and start the slot with the original
workflow. Keep the original node packs installed for comparison.

References: [Modal Image API](https://modal.com/docs/reference/modal.Image),
[GitHub fork visibility](https://docs.github.com/en/pull-requests/reference/forks).
