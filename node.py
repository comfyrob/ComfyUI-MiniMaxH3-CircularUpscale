"""ComfyUI interface; imports the host only when schema/execution needs it."""
import gc
import os

import torch

from .core import load_v1, target_shape, upscale_loop, validate_video

MODEL_FOLDER = "latent_upscale_models"


def model_names():
    import folder_paths
    if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))
    return [n for n in folder_paths.get_filename_list(MODEL_FOLDER)
            if n.lower().endswith((".safetensors", ".pth"))] or ["(install LBH 3D conv v1 checkpoint)"]


class MiniMaxH3CircularLatentUpscale:
    CATEGORY = "video/MinimaxH3"
    FUNCTION = "upscale"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video_latent",)
    DESCRIPTION = ("Loop-aware LBH v1 spatial upscaler. Use on the video stream after "
                   "MiniMax H3 Loop sampling; wraps every temporal convolution and rebuilds "
                   "the two-position tail. Full temporal context, no chunking.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_latent": ("LATENT",),
            "model_name": (model_names(),),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": .05}),
            "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
        }}

    def upscale(self, video_latent, model_name, scale, precision, device):
        import comfy.model_management as mm
        import comfy.utils
        import folder_paths

        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise ValueError("Expected a video LATENT dictionary.")
        # Masks need their own spatial/temporal policy; never silently discard one.
        if video_latent.get("noise_mask") is not None:
            raise ValueError("Masked/inpaint latents are unsupported by this loop adapter.")
        samples = video_latent["samples"]
        body_length = validate_video(samples)
        h_out, w_out = target_shape(samples, scale)
        if precision not in ("fp16", "bf16", "fp32") or device not in ("auto", "cuda", "cpu"):
            raise ValueError("Unsupported precision or device.")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        dev = mm.get_torch_device() if device == "auto" else torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP requested but unavailable; select auto or cpu explicitly.")
        if dev.type == "cuda" and dev.index is None:
            # Comfy's residency manager compares device identities; cuda != cuda:0.
            dev = torch.device("cuda", torch.cuda.current_device())
        if dev.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; choose fp16 or fp32.")
        model = None
        try:
            if (h_out, w_out) != samples.shape[-2:]:
                path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, model_name)
                model = load_v1(comfy.utils.load_torch_file(path, safe_load=True))
                # A full hidden map plus convolution/norm temporaries; a conservative
                # eviction request, not a promise about backend convolution workspace.
                item_bytes = torch.empty((), dtype=dtype).element_size()
                hidden_bytes = samples.shape[0] * 512 * body_length * h_out * w_out * item_bytes
                weight_bytes = sum(p.numel() for p in model.parameters()) * item_bytes
                if dev.type != "cpu":
                    mm.free_memory(weight_bytes + 10 * hidden_bytes, dev)
                model.to(device=dev, dtype=dtype)
            print(f"[H3 Circular Upscale] {body_length} periodic positions + 2 tail; "
                  f"{samples.shape[-1] * 16}x{samples.shape[-2] * 16} -> "
                  f"{w_out * 16}x{h_out * 16}; {precision}; {dev}; circular T, full context")
            out = upscale_loop(samples, model, scale, dtype, dev, mm.throw_exception_if_processing_interrupted)
            result = video_latent.copy()
            result["samples"] = out
            return (result,)
        finally:
            # This node owns its model; no global cache or mutation of the stock node.
            if model is not None:
                model.to(device="cpu")
                del model
            gc.collect()
            if dev.type != "cpu":
                mm.soft_empty_cache()
