The 3D resizer architecture, normalization constants and checkpoint naming
contract are adapted from LBH-123-AI's Comfyui_Minimax_h3_latent_Upscaler,
commit 40316cf008b2fd8663263270669eb4da23f89d2c:
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler

The upstream code is MIT licensed; its license accompanies this package.
Changes: circular temporal convolutions, body-only normalization, explicit
tail reconstruction, strict v1 validation, synchronous output transfer,
finite-value checks and an independent ComfyUI node/model lifecycle.

No model weights are distributed. Their license is separate:
https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

The five-position clip/two-position tail convention interoperates with:
https://github.com/pxl-pshr/ComfyUI-MiniMaxH3-Loop
No source from that repository is vendored.

Native integration (native_backend.py and native_node.py):
The sectioned execution scheme is adapted from ComfyUI's
comfy/ldm/minimax/latent_upscaler.py, introduced by commit
35060f0156181ae78a86b10f098f67ce8e83697f (Kijai / Comfy contributors):
https://github.com/Comfy-Org/ComfyUI
These two files are GPL-3.0-or-later; see LICENSE-native-GPL-3.0.txt.
The original LBH-derived files retain the MIT license in LICENSE. Distributions
including the native integration must retain both notices and license texts.
The native helpers are imported from the host rather than vendored. Changes
include periodic section context, body-only statistics, loop-tail rebuilding,
validation and independent execution without patching the shared model.
