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
