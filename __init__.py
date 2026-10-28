from .node import MiniMaxH3CircularLatentUpscale
from .context import MiniMaxH3LoopContextPad, MiniMaxH3LoopContextRestore

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CircularLatentUpscale": MiniMaxH3CircularLatentUpscale,
    "MiniMaxH3LoopContextPad": MiniMaxH3LoopContextPad,
    "MiniMaxH3LoopContextRestore": MiniMaxH3LoopContextRestore,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CircularLatentUpscale": "MiniMax H3 Circular Latent Upscale",
    "MiniMaxH3LoopContextPad": "MiniMax H3 Loop Context Pad (Experimental)",
    "MiniMaxH3LoopContextRestore": "MiniMax H3 Loop Context Restore (Experimental)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
