# SPDX-License-Identifier: GPL-3.0-or-later
"""Native model-loader integration; keeps the original node and its ID intact."""
import torch

from .core import validate_video, with_tail
from .native_backend import circular_forward, native_api, validate_model


class MiniMaxH3CircularLatentUpscaleWithModel:
    CATEGORY = "video/MinimaxH3"
    FUNCTION = "upscale"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Circular H3 upscale using Load Latent Upscale Model. Processes the loop body "
        "with wraparound temporal context and rebuilds its two-position tail. "
        "Accepts video or joint AV latents; audio passes through. Requires native H3 upscale support."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("LATENT_UPSCALE_MODEL",),
            "samples": ("LATENT",),
            "width": ("INT", {"default": 1920, "min": 32, "max": 16384, "step": 32}),
            "height": ("INT", {"default": 1088, "min": 32, "max": 16384, "step": 32}),
        }}

    @torch.inference_mode()
    def upscale(self, model, samples, width, height):
        import comfy.model_management as mm
        import comfy.nested_tensor
        import comfy.ops

        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Expected a video or joint AV LATENT dictionary.")
        if samples.get("noise_mask") is not None:
            raise ValueError("Masked/inpaint latents are unsupported; the mask will not be discarded.")
        if any(type(v) is not int or v < 32 or v > 16384 or v % 32 for v in (width, height)):
            raise ValueError("width and height must be multiples of 32 between 32 and 16384.")
        latent = samples["samples"]
        joint = isinstance(latent, comfy.nested_tensor.NestedTensor)
        if joint and len(latent.tensors) != 2:
            raise ValueError("Expected a joint H3 latent containing exactly video and audio.")
        video = latent.tensors[0] if joint else latent
        body_length = validate_video(video)
        body = video[:, :, :body_length]
        size = (body_length, height // 16, width // 16)
        mm.throw_exception_if_processing_interrupted()
        if size == tuple(body.shape[2:]):
            output = with_tail(body).to(device=mm.intermediate_device(), copy=True)
        else:
            api = native_api()
            if not all(hasattr(model, attr) for attr in ("model", "load_device", "model_dtype")):
                raise ValueError("Connect the native Load Latent Upscale Model output.")
            validate_model(model.model, api)
            # ComfyUI owns loading/offloading. Never .to(), clone or patch the shared model.
            mm.load_models_gpu([model], memory_required=model.model.memory_required(body.shape, size))
            resized = circular_forward(model.model,
                                       body.to(device=model.load_device, dtype=model.model_dtype()),
                                       size, api, comfy.ops, mm.throw_exception_if_processing_interrupted)
            if not torch.isfinite(resized).all().item():
                raise RuntimeError("Native circular upscaler produced NaN/Inf; output rejected.")
            resized = resized.to(device=mm.intermediate_device(), dtype=video.dtype)
            if not torch.isfinite(resized).all().item():
                raise RuntimeError("Upscaled values overflowed the output latent dtype.")
            output = with_tail(resized)
        result = samples.copy()
        result["samples"] = comfy.nested_tensor.NestedTensor((output, latent.tensors[1])) if joint else output
        print(f"[H3 Circular Upscale With Model] {body_length} periodic positions + 2 tail; "
              f"{video.shape[-1] * 16}x{video.shape[-2] * 16} -> {width}x{height}; "
              "native loader, circular sections, body-wide normalization")
        return (result,)
