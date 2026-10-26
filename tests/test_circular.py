import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from h3_circular.core import (CircularTimeConv3d, LatentResizer3D, checkpoint_state,
                              load_v1, target_shape, upscale_loop, validate_video, with_tail)


def active_model(circular=True):
    torch.manual_seed(813)
    # Same depth and GroupNorm as the real network; reduced channels for CPU tests.
    model = LatentResizer3D(channels=32, circular=circular).eval()
    # Exercise trained residual/temporal paths, which start zeroed in the constructor.
    for layer in model.modules():
        if isinstance(layer, nn.Conv3d):
            nn.init.normal_(layer.weight, std=.018)
            nn.init.normal_(layer.bias, std=.005)
    return model


@pytest.mark.parametrize("kernel,groups", [(3, 1), (5, 2)])
def test_conv_wraps_time_but_not_space(kernel, groups):
    torch.manual_seed(1)
    layer = CircularTimeConv3d(4, 4, (kernel, 3, 3), padding=(kernel // 2, 1, 1), groups=groups)
    x = torch.randn(1, 4, 5, 3, 4)
    pad = kernel // 2
    manual = torch.cat((x[:, :, -pad:], x, x[:, :, :pad]), dim=2)
    expected = F.conv3d(manual, layer.weight, layer.bias, padding=(0, 1, 1), groups=groups)
    torch.testing.assert_close(layer(x), expected)
    torch.testing.assert_close(layer(x.roll(1, 2)), layer(x).roll(1, 2))
    spatially_wrapped = F.conv3d(F.pad(manual, (1, 1, 1, 1, 0, 0), mode="circular"),
                                layer.weight, layer.bias, groups=groups)
    assert not torch.allclose(expected, spatially_wrapped)


def test_all_temporal_layers_are_circular():
    with torch.device("meta"):
        model = LatentResizer3D()
    layers = [m for m in model.modules() if isinstance(m, nn.Conv3d) and m.kernel_size[0] > 1]
    assert len(layers) == 62
    assert all(isinstance(m, CircularTimeConv3d) for m in layers)


def test_full_depth_network_is_equivariant_to_whole_clip_rotation():
    model = active_model()
    body = torch.randn(1, 24, 45, 2, 3)
    with torch.inference_mode():
        y = model(body, 2., (45, 4, 6))
        shifted = model(body.roll(5, 2), 2., (45, 4, 6))
    torch.testing.assert_close(shifted, y.roll(5, 2), atol=3e-5, rtol=3e-5)
    # This assertion fails the property for the old boundary treatment using identical weights.
    stock = active_model(circular=False)
    stock.load_state_dict(model.state_dict())
    with torch.inference_mode():
        old = stock(body, 2., (45, 4, 6))
        old_shifted = stock(body.roll(5, 2), 2., (45, 4, 6))
    assert (old_shifted - old.roll(5, 2)).abs().max() > .01


def test_constant_loop_has_no_artificial_temporal_endpoints():
    model = active_model()
    body = torch.randn(1, 24, 1, 2, 3).expand(-1, -1, 5, -1, -1)
    with torch.inference_mode():
        y = model(body, 2., (5, 4, 6))
    torch.testing.assert_close(y, y[:, :, :1].expand_as(y), atol=3e-5, rtol=3e-5)


def test_tail_is_excluded_from_network_and_rebuilt_without_mutating_input():
    model = active_model()
    body = torch.randn(2, 24, 5, 2, 3)
    a = with_tail(body)
    b = a.clone()
    b[:, :, -2:] = 1234
    before = b.clone()
    out_a = upscale_loop(a, model, 2., torch.float32, torch.device("cpu"))
    out_b = upscale_loop(b, model, 2., torch.float32, torch.device("cpu"))
    assert out_a.shape == (2, 24, 7, 4, 6)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    torch.testing.assert_close(out_b[:, :, -2:], out_b[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(b, before, rtol=0, atol=0)
    assert out_b.device.type == "cpu" and not out_b.requires_grad


def test_no_resize_skips_model_but_canonicalizes_tail():
    samples = torch.randn(1, 24, 7, 3, 3)
    out = upscale_loop(samples, None, 1., torch.float32, torch.device("cpu"))
    torch.testing.assert_close(out[:, :, :-2], samples[:, :, :-2], rtol=0, atol=0)
    torch.testing.assert_close(out[:, :, -2:], samples[:, :, :2], rtol=0, atol=0)
    assert out.data_ptr() != samples.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_selected_compute_precision_and_output_dtype(dtype):
    model = active_model().to(dtype=dtype)
    x = torch.randn(1, 24, 7, 2, 3, dtype=torch.float32)
    out = upscale_loop(x, model, 2., dtype, torch.device("cpu"))
    assert out.dtype == x.dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :, -2:], out[:, :, :2], rtol=0, atol=0)


@pytest.mark.parametrize("samples", [torch.zeros(1, 24, 6, 2, 2), torch.zeros(1, 24, 2, 2, 2),
                                    torch.zeros(1, 16, 7, 2, 2), torch.zeros(1, 24, 2, 2),
                                    torch.zeros(0, 24, 7, 2, 2), object()])
def test_rejects_invalid_or_joint_latents(samples):
    with pytest.raises(ValueError):
        validate_video(samples)


def test_rejects_nonfinite_input_and_output():
    x = torch.ones(1, 24, 7, 2, 2)
    x[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_video(x)
    x.nan_to_num_()

    def broken(x, scale, size, check_interrupt):
        return torch.full((x.shape[0], 24, *size), torch.inf)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        upscale_loop(x, broken, 2., torch.float32, torch.device("cpu"))


def test_rejects_output_dtype_overflow():
    x = torch.ones(1, 24, 7, 2, 2, dtype=torch.float16)

    def huge(x, scale, size, check_interrupt):
        return torch.full((1, 24, *size), 1e10)
    with pytest.raises(RuntimeError, match="overflowed"):
        upscale_loop(x, huge, 2., torch.float32, torch.device("cpu"))


def test_requested_dimensions_and_scale_guards():
    x = torch.empty(1, 24, 47, 34, 60, device="meta")
    assert target_shape(x, 2.) == (68, 120)  # 1920 x 1088
    for scale in (0., float("nan"), float("inf"), 4.1):
        with pytest.raises(ValueError):
            target_shape(x, scale)


def test_published_checkpoint_header_matches_every_parameter():
    header = json.loads((Path(__file__).parent / "fixtures/v1-bf16-header.json").read_text())
    with torch.device("meta"):
        state = LatentResizer3D().state_dict()
    assert set(state) == set(header)
    assert len(state) == 322
    assert sum(v.numel() for v in state.values()) == 345280216
    assert all(list(state[k].shape) == h["shape"] for k, h in header.items())


def test_checkpoint_loader_is_strict_and_preserves_all_keys():
    with torch.device("meta"):
        state = LatentResizer3D().state_dict()
    assert load_v1(state).conv_in.weight.device.type == "meta"
    assert set(checkpoint_state({"model": {"upscaler." + k: v for k, v in state.items()}})) == set(state)
    for bad in ({**state, "extra": torch.empty(1)},
                {k: v for k, v in state.items() if k != "conv_in.bias"},
                {**state, "conv_in.bias": torch.empty(2)}):
        with pytest.raises(ValueError, match="supported LBH"):
            load_v1(bad)
    with pytest.raises(ValueError, match="Mixed"):
        checkpoint_state({"upscaler.x": torch.empty(1), "y": torch.empty(1)})


def test_interrupt_propagates_during_network_execution():
    model = active_model()
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        upscale_loop(torch.randn(1, 24, 7, 2, 3), model, 2., torch.float32,
                     torch.device("cpu"), interrupt)
    assert calls == 3
