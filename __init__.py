from .node import MiniMaxH3CircularLatentUpscale
from .native_node import MiniMaxH3CircularLatentUpscaleWithModel

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CircularLatentUpscale": MiniMaxH3CircularLatentUpscale,
    "MiniMaxH3CircularLatentUpscaleWithModel": MiniMaxH3CircularLatentUpscaleWithModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CircularLatentUpscale": "MiniMax H3 Circular Latent Upscale",
    "MiniMaxH3CircularLatentUpscaleWithModel": "MiniMax H3 Circular Latent Upscale With Model",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
