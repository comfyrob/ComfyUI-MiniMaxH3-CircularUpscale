"""LBH v1 latent-resizer architecture with periodic temporal boundaries.

Architecture adapted from LBH-123-AI, MIT; see LICENSE and NOTICE.md.
Checkpoint parameter names and the upstream normalization convention are kept.
No ComfyUI imports: the boundary behavior can be tested independently.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

CLIP_LATENTS = 5
TAIL_LATENTS = 2
LATENTS_MEAN = [
    .858090341091156, -.9606591463088989, 1.0661640167236328, -.5090325474739075,
    -.2727581858634949, -1.3675414323806763, -.2553254961967468, -.26907554268836975,
    -.5376840829849243, -.0464097298681736, .6657370328903198, .19690127670764923,
    -.5460608005523682, -.4035342037677765, -.23683024942874908, .25928452610969543,
    -.30133944749832153, .211341992020607, -1.1206848621368408, .3581933379173279,
    -.04225143790245056, .2604829967021942, .22864092886447906, .7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, .9653137922286987, 1.0569885969161987,
    .841948926448822, .7729952931404114, 1.8955937623977661, .946841835975647,
    .7996809482574463, .44988900423049927, .7197399735450745, .6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


class CircularTimeConv3d(nn.Conv3d):
    """Wrap only T; preserve zero padding in H/W and unchanged parameter names."""

    def forward(self, x):
        pt, ph, pw = self.padding
        if pt:
            # Modulo indexing also works when a loop is shorter than its padding.
            indices = torch.arange(-pt, x.shape[2] + pt, device=x.device) % x.shape[2]
            x = x.index_select(2, indices)
        return F.conv3d(x, self.weight, self.bias, self.stride,
                        (0, ph, pw), self.dilation, self.groups)


def _zero(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, conv, dropout=.1):
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(),
                                       conv(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(64, 2 * channels))
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout),
                                        _zero(conv(channels, channels, 3, padding=1)))
        self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype)[..., None, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels, conv):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.dwconv = conv(channels, channels, (5, 1, 1), padding=(2, 0, 0), groups=channels)
        self.pwconv = _zero(conv(channels, channels, 1))

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    """Released v1 backbone. Smaller channel/block counts are for unit tests only."""

    def __init__(self, channels=512, in_blocks=12, out_blocks=12, circular=True):
        super().__init__()
        conv = CircularTimeConv3d if circular else nn.Conv3d
        self.conv_in = conv(24, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

        def stage(count):
            layers = []
            for i in range(count):
                layers.append(ResBlockEmb3D(channels, conv))
                if i % 2 == 0:
                    layers.append(TemporalConv(channels, conv))
            return nn.ModuleList(layers)

        self.in_blocks = stage(in_blocks)
        self.out_blocks = stage(out_blocks)
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = conv(channels, 24, 3, padding=1)

    def forward(self, x, scale, target_size, check_interrupt=lambda: None):
        if target_size[0] != x.shape[2]:
            raise ValueError("Circular H3 upscale must not resize time.")
        emb = self.embed(x.new_tensor([[scale - 1]])).expand(x.shape[0], -1)
        x = self.conv_in(x)
        for stage in (self.in_blocks, self.out_blocks):
            if stage is self.out_blocks:
                x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
            for block in stage:
                check_interrupt()
                x = block(x, emb) if isinstance(block, ResBlockEmb3D) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def validate_video(samples):
    if not isinstance(samples, torch.Tensor) or samples.is_nested:
        raise ValueError("Connect the video_latent output of Separate AV Latent, not the joint AV latent.")
    if samples.ndim != 5 or samples.shape[1] != 24 or not samples.is_floating_point():
        raise ValueError("Expected floating H3 video samples shaped [B,24,T,H,W].")
    if min(samples.shape) < 1:
        raise ValueError("Empty latent dimensions are not supported.")
    body_length = samples.shape[2] - TAIL_LATENTS
    if body_length < CLIP_LATENTS or body_length % CLIP_LATENTS:
        raise ValueError("A loop must have 5*k+2 latent positions (17*k+5 requested video frames).")
    if not torch.isfinite(samples).all().item():
        raise ValueError("Incoming latent contains NaN/Inf; circular padding cannot repair it.")
    return body_length


def target_shape(samples, scale):
    if not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError("scale must be between 1 and 4.")
    h, w = samples.shape[-2:]
    # Match upstream scale mode with align=32. Two latent positions = 32 pixels.
    if scale == 1:
        return h, w
    return max(2, round(h * scale / 2) * 2), max(2, round(w * scale / 2) * 2)


def with_tail(body):
    return torch.cat((body, body[:, :, :TAIL_LATENTS]), dim=2)


@torch.inference_mode()
def upscale_loop(samples, model, scale, dtype, device, check_interrupt=lambda: None):
    """No input mutation; output is synchronously copied to CPU, in input dtype."""
    body_length = validate_video(samples)
    h_out, w_out = target_shape(samples, scale)
    body = samples[:, :, :body_length]
    if (h_out, w_out) == samples.shape[-2:]:
        return with_tail(body).to(device="cpu", copy=True)
    if model is None:
        raise ValueError("An upscale model is required when spatial dimensions change.")
    mean = torch.tensor(LATENTS_MEAN, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    body = body.to(device=device, dtype=dtype)
    # Preserve LBH's trained/inference convention; do not speculate about removing it.
    normalized = (body - mean) / std
    check_interrupt()
    out = model(normalized, scale, (body_length, h_out, w_out), check_interrupt)
    out = out * std + mean
    if not torch.isfinite(out).all().item():
        raise RuntimeError("Upscaler produced NaN/Inf; no output was passed to the VAE.")
    out = out.to(device="cpu", dtype=samples.dtype)  # blocking: downstream CPU reads are safe
    if not torch.isfinite(out).all().item():
        raise RuntimeError("Upscaled values overflowed the output latent dtype.")
    return with_tail(out)


def checkpoint_state(raw):
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        raw = raw["model"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty v1 upscaler state dictionary.")
    prefixed = [isinstance(k, str) and k.startswith("upscaler.") for k in raw]
    if any(prefixed):
        if not all(prefixed):
            raise ValueError("Mixed upscaler prefixes; refusing to discard checkpoint entries.")
        raw = {k[len("upscaler."):]: v for k, v in raw.items()}
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) or not v.is_floating_point()
           for k, v in raw.items()):
        raise ValueError("Only floating-point v1 upscaler state dictionaries are supported.")
    return raw


def load_v1(raw):
    """Validate all names/shapes before allocating device memory; no silent missing keys."""
    state = checkpoint_state(raw)
    with torch.device("meta"):
        model = LatentResizer3D()
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    wrong = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or wrong:
        raise ValueError("Checkpoint is not the supported LBH 3D conv v1 architecture. "
                         f"Missing: {missing[:4]}; unexpected: {unexpected[:4]}; wrong shapes: {wrong[:4]}")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)
