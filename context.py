"""Experimental periodic context for external H3 video-latent processors.

This pads a processor's input once; it cannot change its internal temporal
padding, normalization, chunking, or denoising steps. See docs/LOOP-CONTEXT.md.
"""
from dataclasses import dataclass

import torch

from .core import CLIP_LATENTS, TAIL_LATENTS, validate_video, with_tail


@dataclass(frozen=True)
class LoopContext:
    body_length: int
    pad_latents: int
    batch_size: int

    @property
    def padded_length(self):
        return self.body_length + 2 * self.pad_latents + TAIL_LATENTS


def video_samples(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a separate H3 video LATENT dictionary.")
    if latent.get("noise_mask") is not None:
        raise ValueError("Masked/inpaint latents need a separate context policy and are unsupported.")
    samples = latent["samples"]
    validate_video(samples)
    return samples


@torch.no_grad()
def pad_context(video_latent, pad_clips):
    if type(pad_clips) is not int or not 1 <= pad_clips <= 64:
        raise ValueError("pad_clips must be an integer between 1 and 64.")
    samples = video_samples(video_latent)
    body_length = samples.shape[2] - TAIL_LATENTS
    context = LoopContext(body_length, pad_clips * CLIP_LATENTS, samples.shape[0])
    # Remove the bookkeeping tail before wrapping, even if it is stale.
    indices = torch.arange(-context.pad_latents,
                           body_length + context.pad_latents + TAIL_LATENTS,
                           device=samples.device) % body_length
    result = video_latent.copy()
    result["samples"] = samples[:, :, :body_length].index_select(2, indices)
    return result, context


@torch.no_grad()
def restore_context(video_latent, context):
    if not isinstance(context, LoopContext):
        raise ValueError("Connect loop_context from the matching MiniMax H3 Loop Context Pad node.")
    samples = video_samples(video_latent)
    if samples.shape[0] != context.batch_size or samples.shape[2] != context.padded_length:
        raise ValueError(
            "The processor must preserve batch size and temporal length. "
            f"Expected B={context.batch_size}, T={context.padded_length}; "
            f"got B={samples.shape[0]}, T={samples.shape[2]}. "
            "Use the matching Pad context and disable frame trimming or temporal resizing."
        )
    start = context.pad_latents
    body = samples[:, :, start:start + context.body_length]
    result = video_latent.copy()
    result["samples"] = with_tail(body)
    return result


class MiniMaxH3LoopContextPad:
    CATEGORY = "video/MinimaxH3/experimental"
    FUNCTION = "pad"
    RETURN_TYPES = ("LATENT", "H3_LOOP_CONTEXT")
    RETURN_NAMES = ("padded_video_latent", "loop_context")
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Experimental: surround a separate H3 video latent with periodic context before an external "
        "processor. Connect its result and this context to Loop Context Restore. "
        "Does not replace circular convolutions or per-step Loop sampling. Full temporal processing "
        "is required for the LBH experiment; extra context increases memory use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_latent": ("LATENT", {"tooltip": "Separate H3 video stream, shaped [B,24,5*k+2,H,W]."}),
            "pad_clips": ("INT", {"default": 3, "min": 1, "max": 64,
                                  "tooltip": "Each clip adds 5 latent positions on each side. "
                                             "3 is a small-context experiment; 15 covers the LBH v1 "
                                             "convolutional radius but does not eliminate GroupNorm effects."}),
        }}

    def pad(self, video_latent, pad_clips):
        return pad_context(video_latent, pad_clips)


class MiniMaxH3LoopContextRestore:
    CATEGORY = "video/MinimaxH3/experimental"
    FUNCTION = "restore"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video_latent",)
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Crop the processed context back to the original loop body and rebuild its two-position tail. "
        "Allows spatial resizing; rejects changed batch size, temporal length, masks, or nonfinite values. "
        "The processor must preserve temporal order. Tail consistency does not guarantee a seamless video."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_latent": ("LATENT",),
            "loop_context": ("H3_LOOP_CONTEXT",),
        }}

    def restore(self, video_latent, loop_context):
        return (restore_context(video_latent, loop_context),)
