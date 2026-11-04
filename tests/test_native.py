"""CPU reference checks; these do not benchmark CUDA/comfy-kitchen kernels."""
import sys
import types

import pytest
import torch
from torch.nn import functional as F

from h3_circular import NODE_CLASS_MAPPINGS, native_backend, native_node
from h3_circular.core import (LATENTS_MEAN, LATENTS_STD, LatentResizer3D,
                              ResBlockEmb3D, TemporalConv, upscale_loop, with_tail)


class NativeModel(LatentResizer3D):
    """The native architecture has identical checkpoint keys; helpers run its layers."""
    def memory_required(self, shape, size):
        return shape[0] * self.conv_in.out_channels * (2 * shape[2] + 96) * max(
            shape[-2] * shape[-1], size[-2] * size[-1]) * self.conv_in.weight.element_size()


def reference_stats(x, norm):
    groups = x.float().reshape(x.shape[0], norm.num_groups, -1)
    variance, mean = torch.var_mean(groups, dim=-1, correction=0, keepdim=True)
    return mean, torch.rsqrt(variance + norm.eps)


def reference_norm(x, norm, stats, scale=None, shift=None):
    mean, inverse_std = stats
    normalized = ((x.float().reshape(x.shape[0], norm.num_groups, -1) - mean)
                  * inverse_std).reshape(x.shape)
    weight, bias = norm.weight, norm.bias
    if scale is not None:
        weight = weight * (1 + scale)
        bias = bias * (1 + scale) + shift
    y = normalized * weight.reshape(1, -1, 1, 1, 1) + bias.reshape(1, -1, 1, 1, 1)
    return F.silu(y).to(x.dtype).contiguous(memory_format=torch.channels_last_3d)


def reference_conv(conv, x):
    return F.conv3d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


@pytest.fixture
def native_host(monkeypatch):
    calls = []
    api = types.SimpleNamespace(MiniMaxH3LatentUpscaler=NativeModel,
                                ResBlockEmb3D=ResBlockEmb3D, TemporalConv=TemporalConv,
                                LATENTS_MEAN=LATENTS_MEAN, LATENTS_STD=LATENTS_STD,
                                clip_stats=reference_stats, norm_silu=reference_norm,
                                conv3d_cl=reference_conv)
    ops = types.ModuleType("comfy.ops")

    def cast(layer, x, offloadable):
        assert offloadable
        calls.append(("cast", id(layer)))
        return layer.weight, layer.bias, id(layer)

    def uncast(layer, weight, bias, lease):
        assert lease == id(layer)
        calls.append(("uncast", id(layer)))

    ops.cast_bias_weight, ops.uncast_bias_weight = cast, uncast
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda models, memory_required: calls.append(("load", memory_required))
    mm.intermediate_device = lambda: torch.device("cpu")
    mm.throw_exception_if_processing_interrupted = lambda: None
    nested = types.ModuleType("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True

        def __init__(self, tensors):
            self.tensors = list(tensors)

    nested.NestedTensor = NestedTensor
    comfy.model_management, comfy.ops, comfy.nested_tensor = mm, ops, nested
    for module in (comfy, mm, ops, nested):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(native_node, "native_api", lambda: api)
    return types.SimpleNamespace(api=api, ops=ops, mm=mm, nested=NestedTensor, calls=calls)


def models(depth=12, dtype=torch.float32):
    torch.manual_seed(401)
    native = NativeModel(channels=32, in_blocks=depth, out_blocks=depth, circular=False).eval()
    for layer in native.modules():
        if isinstance(layer, torch.nn.Conv3d):
            torch.nn.init.normal_(layer.weight, std=.018)
            torch.nn.init.normal_(layer.bias, std=.005)
    circular = LatentResizer3D(channels=32, in_blocks=depth, out_blocks=depth).eval()
    circular.load_state_dict(native.state_dict())
    return native.to(dtype), circular.to(dtype)


def patcher(model):
    return types.SimpleNamespace(model=model, load_device=torch.device("cpu"),
                                 model_dtype=lambda: model.conv_in.weight.dtype)


@pytest.mark.parametrize("length,slab,batch", [(5, 16, 1), (20, 16, 2), (45, 16, 1), (5, 1, 1)])
def test_native_sections_match_existing_full_depth_circular_model(native_host, length, slab, batch):
    host = native_host
    native, reference = models()
    x = torch.randn(batch, 24, length, 2, 3)
    before = x.clone()
    state = {key: value.clone() for key, value in native.state_dict().items()}
    forward = native.forward.__func__
    native_backend.validate_model(native, host.api)
    actual = native_backend.circular_forward(native, x, (length, 4, 6), host.api, host.ops, slab=slab)
    expected = upscale_loop(with_tail(x), reference, 2., torch.float32, torch.device("cpu"))[:, :, :-2]
    torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-5)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert native.forward.__func__ is forward
    for key, value in native.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert [c[1] for c in host.calls if c[0] == "cast"] == [c[1] for c in host.calls if c[0] == "uncast"]


def test_rotations_across_section_and_loop_boundaries(native_host):
    host = native_host
    native, _ = models()
    x = torch.randn(1, 24, 45, 2, 3)
    expected = native_backend.circular_forward(native, x, (45, 4, 6), host.api, host.ops)
    for shift in (1, 5, 16, 44):
        actual = native_backend.circular_forward(native, x.roll(shift, 2), (45, 4, 6), host.api, host.ops)
        torch.testing.assert_close(actual, expected.roll(shift, 2), atol=8e-5, rtol=8e-5)


def test_constant_loop_has_no_section_or_endpoint_flicker(native_host):
    host = native_host
    native, _ = models()
    x = torch.randn(1, 24, 1, 2, 3).expand(-1, -1, 20, -1, -1)
    actual = native_backend.circular_forward(native, x, (20, 4, 6), host.api, host.ops)
    torch.testing.assert_close(actual, actual[:, :, :1].expand_as(actual), atol=8e-5, rtol=8e-5)


def test_joint_node_preserves_audio_metadata_and_shared_model(native_host):
    host = native_host
    native, reference = models()
    video = with_tail(torch.randn(2, 24, 20, 2, 4))
    video[:, :, -2:] = 900  # Tail must affect neither convolution nor GroupNorm.
    before = video.clone()
    audio = torch.randn(2, 32, 2, 100)
    source = {"samples": host.nested((video, audio)), "batch_index": [0, 1]}
    out, = native_node.MiniMaxH3CircularLatentUpscaleWithModel().upscale(patcher(native), source, 128, 64)
    result = out["samples"].tensors[0]
    expected = upscale_loop(video, reference, 2., torch.float32, torch.device("cpu"))
    torch.testing.assert_close(result, expected, atol=8e-5, rtol=8e-5)
    torch.testing.assert_close(result[:, :, -2:], result[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(video, before, rtol=0, atol=0)
    assert out["samples"].tensors[1] is audio
    assert out["batch_index"] is source["batch_index"]
    assert out is not source and out["samples"] is not source["samples"]
    assert host.calls[0] == ("load", native.memory_required((2, 24, 20, 2, 4), (20, 4, 8)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_precision_and_output_dtype(native_host, dtype):
    host = native_host
    native, _ = models(depth=2, dtype=dtype)
    video = torch.randn(1, 24, 7, 2, 4, dtype=torch.float32)
    result, = native_node.MiniMaxH3CircularLatentUpscaleWithModel().upscale(patcher(native), {"samples": video}, 128, 64)
    assert result["samples"].dtype == video.dtype
    assert torch.isfinite(result["samples"]).all()
    torch.testing.assert_close(result["samples"][:, :, -2:], result["samples"][:, :, :2], rtol=0, atol=0)


def test_no_resize_repairs_tail_without_loading_or_changing_audio(native_host):
    host = native_host
    video = torch.randn(1, 24, 7, 2, 4)
    audio = torch.randn(1, 32, 2, 10)
    out, = native_node.MiniMaxH3CircularLatentUpscaleWithModel().upscale(None, {"samples": host.nested((video, audio))}, 64, 32)
    assert host.calls == []
    assert out["samples"].tensors[1] is audio
    result = out["samples"].tensors[0]
    assert result.data_ptr() != video.data_ptr()
    torch.testing.assert_close(result[:, :, :-2], video[:, :, :-2], rtol=0, atol=0)
    torch.testing.assert_close(result[:, :, -2:], video[:, :, :2], rtol=0, atol=0)


def test_rejects_masks_bad_dimensions_and_other_model_families_before_loading(native_host):
    host = native_host
    node = native_node.MiniMaxH3CircularLatentUpscaleWithModel()
    samples = {"samples": torch.randn(1, 24, 7, 2, 4)}
    with pytest.raises(ValueError, match="Masked"):
        node.upscale(None, {**samples, "noise_mask": torch.ones(1)}, 128, 64)
    with pytest.raises(ValueError, match="multiples of 32"):
        node.upscale(None, samples, 127, 64)
    with pytest.raises(ValueError, match="H3 3D-conv"):
        node.upscale(patcher(torch.nn.Identity()), samples, 128, 64)
    with pytest.raises(ValueError, match="exactly video and audio"):
        node.upscale(None, {"samples": host.nested([samples["samples"]])}, 128, 64)
    assert host.calls == []


def test_cancellation_releases_temporal_weight_lease(native_host):
    host = native_host
    native, _ = models(depth=2)
    state = {k: v.clone() for k, v in native.state_dict().items()}

    def cancel_inside_temporal():
        if any(c[0] == "cast" for c in host.calls):
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        native_backend.circular_forward(native, torch.randn(1, 24, 5, 2, 4), (5, 4, 8),
                                         host.api, host.ops, cancel_inside_temporal)
    assert [c[0] for c in host.calls] == ["cast", "uncast"]
    for key, value in native.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("value,dtype,message", [(float("nan"), torch.float32, "NaN/Inf"),
                                                (1e10, torch.float16, "overflowed")])
def test_rejects_bad_output(native_host, monkeypatch, value, dtype, message):
    native, _ = models(depth=2)
    monkeypatch.setattr(native_node, "circular_forward", lambda *args: torch.full((1, 24, 5, 4, 8), value))
    with pytest.raises(RuntimeError, match=message):
        native_node.MiniMaxH3CircularLatentUpscaleWithModel().upscale(
            patcher(native), {"samples": torch.randn(1, 24, 7, 2, 4, dtype=dtype)}, 128, 64)


def test_registration_retains_old_node_and_uses_native_socket():
    assert "MiniMaxH3CircularLatentUpscale" in NODE_CLASS_MAPPINGS
    new = NODE_CLASS_MAPPINGS["MiniMaxH3CircularLatentUpscaleWithModel"]
    assert new.INPUT_TYPES()["required"]["model"] == ("LATENT_UPSCALE_MODEL",)
