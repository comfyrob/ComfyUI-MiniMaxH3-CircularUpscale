# SPDX-License-Identifier: GPL-3.0-or-later
"""Circular execution of a ComfyUI-owned H3 upscaler, without model patching.

Uses the host's normalization, convolution and weight-offload helpers. Only
activation tensors owned by this invocation are modified. The execution scheme
is adapted from ComfyUI's minimax/latent_upscaler.py; see NOTICE.md.
"""
import torch
from torch.nn import functional as F


def native_api():
    try:
        from comfy.ldm.minimax import latent_upscaler
    except ImportError as exc:
        raise RuntimeError(
            "This node requires ComfyUI with native MiniMax H3 latent upscale support "
            "(commit 35060f0 or newer) and its comfy-kitchen dependency."
        ) from exc
    required = ("MiniMaxH3LatentUpscaler", "ResBlockEmb3D", "TemporalConv",
                "clip_stats", "norm_silu", "conv3d_cl", "LATENTS_MEAN", "LATENTS_STD")
    if not all(hasattr(latent_upscaler, key) for key in required):
        raise RuntimeError("The installed native H3 upscaler API is incompatible with this adapter.")
    return latent_upscaler


def validate_model(model, api):
    # Other families use the same socket type, but different latent conventions.
    if type(model) is not api.MiniMaxH3LatentUpscaler:
        raise ValueError("Connect Load Latent Upscale Model with an H3 3D-conv checkpoint.")
    if model.conv_in.in_channels != 24 or model.conv_out.out_channels != 24:
        raise ValueError("This circular adapter requires the 24-channel H3 video architecture.")
    for block in (*model.in_blocks, *model.out_blocks):
        if type(block) not in (api.ResBlockEmb3D, api.TemporalConv):
            raise ValueError("Unsupported block in the native H3 upscaler.")
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv3d):
            if (layer.stride != (1, 1, 1) or layer.dilation != (1, 1, 1)
                    or layer.padding_mode != "zeros"
                    or any(k % 2 != 1 or p != k // 2
                           for k, p in zip(layer.kernel_size, layer.padding))):
                raise ValueError("Unsupported native convolution geometry; circular execution stopped.")


def _context(x, start, stop, halo):
    """Gather wraparound neighbors; use a view for interior sections."""
    if start >= halo and stop + halo <= x.shape[2]:
        return x[:, :, start - halo:stop + halo]
    indices = torch.arange(start - halo, stop + halo, device=x.device) % x.shape[2]
    return x.index_select(2, indices)


def _sections(length, slab):
    for start in range(0, length, slab):
        yield start, min(start + slab, length)


def _convolve(api, conv, x, slab, interrupt, norm=None, modulation=(), add_to=None):
    """Normalize once over the body, then convolve sections with circular halos."""
    stats = api.clip_stats(x, norm) if norm is not None else None
    result = add_to
    if result is None:
        result = torch.empty((x.shape[0], conv.out_channels, *x.shape[2:]),
                             device=x.device, dtype=x.dtype,
                             memory_format=torch.channels_last_3d)
    halo = conv.padding[0]
    for start, stop in _sections(x.shape[2], slab):
        interrupt()
        section = _context(x, start, stop, halo)
        if norm is not None:
            section = api.norm_silu(section, norm, stats, *modulation)
        # Native zero padding affects only the extra halo outputs, which we drop.
        values = api.conv3d_cl(conv, section)[:, :, halo:halo + stop - start]
        if add_to is None:
            result[:, :, start:stop].copy_(values)
        else:
            result[:, :, start:stop].add_(values)
    return result


def _temporal(api, ops, block, x, slab, interrupt):
    """Depthwise time filtering without cuDNN's slow channels-last depthwise path."""
    stats = api.clip_stats(x, block.norm)
    result = torch.empty_like(x)
    weight, bias, lease = ops.cast_bias_weight(block.dwconv, x, offloadable=True)
    try:
        kernel = weight.shape[2]
        for start, stop in _sections(x.shape[2], slab):
            interrupt()
            section = api.norm_silu(_context(x, start, stop, kernel // 2), block.norm, stats)
            filtered = torch.zeros_like(x[:, :, start:stop])
            if bias is not None:
                filtered.copy_(bias.reshape(1, -1, 1, 1, 1))
            for offset in range(kernel):
                filtered.addcmul_(section[:, :, offset:offset + stop - start],
                                 weight[:, 0, offset].reshape(1, -1, 1, 1, 1))
            result[:, :, start:stop] = api.conv3d_cl(block.pwconv, filtered)
    finally:
        ops.uncast_bias_weight(block.dwconv, weight, bias, lease)
    # Delay the residual update until all wraparound reads of x have finished.
    return x.add_(result)


@torch.inference_mode()
def circular_forward(model, body, size, api, ops, interrupt=lambda: None, slab=16):
    """Native normalized H3 body -> resized body; no tail and no time resizing.

    Slab size is an implementation detail, not a temporal-context limit.
    The caller validates the model and manages its residency through ComfyUI.
    """
    if slab < 1 or size[0] != body.shape[2]:
        raise ValueError("Circular execution requires a positive slab size and unchanged time.")
    interrupt()
    mean = body.new_tensor(api.LATENTS_MEAN).reshape(1, 24, 1, 1, 1)
    std = body.new_tensor(api.LATENTS_STD).reshape(1, 24, 1, 1, 1)
    scale = (size[-1] / body.shape[-1] + size[-2] / body.shape[-2]) / 2
    embedding = model.embed(body.new_tensor([[scale - 1]]))
    # Arithmetic allocates a private tensor, even when body.to() returned a view.
    x = _convolve(api, model.conv_in, (body - mean) / std, slab, interrupt)
    for stage_index, blocks in enumerate((model.in_blocks, model.out_blocks)):
        if stage_index:
            interrupt()
            x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
            x = x.contiguous(memory_format=torch.channels_last_3d)
        for block in blocks:
            interrupt()
            if type(block) is api.ResBlockEmb3D:
                modulation = block.emb_layers(embedding)[0].chunk(2)
                hidden = _convolve(api, block.in_layers[2], x, slab, interrupt,
                                   norm=block.in_layers[0])
                x = _convolve(api, block.out_layers[2], hidden, slab, interrupt,
                              norm=block.out_norm, modulation=modulation, add_to=x)
                del hidden
            else:
                x = _temporal(api, ops, block, x, slab, interrupt)
    output = _convolve(api, model.conv_out, x, slab, interrupt, norm=model.norm_out)
    return torch.addcmul(mean, output, std)
