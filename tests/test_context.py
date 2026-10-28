import pytest
import torch
from torch.nn import functional as F

from h3_circular import NODE_CLASS_MAPPINGS
from h3_circular.context import (LoopContext, MiniMaxH3LoopContextPad,
                                 MiniMaxH3LoopContextRestore, pad_context, restore_context)
from h3_circular.core import with_tail


def latent(body_length=15, batch=1):
    body = torch.arange(body_length, dtype=torch.float32).view(1, 1, -1, 1, 1)
    return {"samples": with_tail(body.expand(batch, 24, -1, 2, 3).clone()),
            "batch_index": list(range(batch))}


@pytest.mark.parametrize("body_length,pad_clips", [(5, 3), (15, 1), (45, 15)])
def test_identity_processor_restores_original_loop_and_discards_stale_tail(body_length, pad_clips):
    source = latent(body_length, batch=2)
    source["samples"][:, :, -2:] = -999
    before = source["samples"].clone()
    padded, context = pad_context(source, pad_clips)
    expected = torch.arange(-5 * pad_clips, body_length + 5 * pad_clips + 2) % body_length
    torch.testing.assert_close(padded["samples"][0, 0, :, 0, 0], expected.float())
    result = restore_context(padded, context)
    torch.testing.assert_close(result["samples"], with_tail(before[:, :, :-2]))
    torch.testing.assert_close(source["samples"], before)
    assert result["batch_index"] is source["batch_index"]
    assert result is not padded and padded is not source


def test_external_processor_can_resize_space_and_return_fresh_metadata():
    padded, context = pad_context(latent(), 3)
    processed = {"samples": F.interpolate(padded["samples"], size=(context.padded_length, 4, 6)),
                 "processor_note": "new"}
    result = restore_context(processed, context)
    assert result["samples"].shape == (1, 24, 17, 4, 6)
    assert result["processor_note"] == "new"
    torch.testing.assert_close(result["samples"][:, :, -2:], result["samples"][:, :, :2])
    assert result["samples"].data_ptr() != processed["samples"].data_ptr()


def test_pad_and_restore_match_circular_single_convolution_when_context_is_sufficient():
    torch.manual_seed(92)
    source = {"samples": with_tail(torch.randn(1, 24, 15, 2, 3))}
    padded, context = pad_context(source, 1)
    weight = torch.randn(24, 1, 5, 1, 1)
    processed = {"samples": F.conv3d(padded["samples"], weight, padding=(2, 0, 0), groups=24)}
    result = restore_context(processed, context)
    body = source["samples"][:, :, :-2]
    periodic = torch.cat((body[:, :, -2:], body, body[:, :, :2]), dim=2)
    expected = with_tail(F.conv3d(periodic, weight, groups=24))
    torch.testing.assert_close(result["samples"], expected)


def test_groupnorm_counterexample_padding_is_not_a_universal_loop_fix():
    # Even beyond a convolution's radius, its edge errors affect global stats.
    torch.manual_seed(19)
    source = {"samples": with_tail(torch.randn(1, 24, 15, 2, 3))}
    weight = torch.randn(24, 1, 3, 1, 1)

    def run(samples):
        padded, context = pad_context({"samples": samples}, 3)
        output = F.conv3d(padded["samples"], weight, padding=(1, 0, 0), groups=24)
        output = F.group_norm(output, 24)
        return restore_context({"samples": output}, context)["samples"][:, :, :-2]

    body = source["samples"][:, :, :-2]
    y = run(source["samples"])
    shifted = run(with_tail(body.roll(5, 2))).roll(-5, 2)
    assert (y - shifted).abs().max() > 1e-3


@pytest.mark.parametrize("pad_clips", [0, 65, -1, True, 1.5])
def test_reject_invalid_padding(pad_clips):
    with pytest.raises(ValueError, match="pad_clips"):
        pad_context(latent(), pad_clips)


def test_reject_masked_nonfinite_and_joint_inputs():
    with pytest.raises(ValueError, match="Masked"):
        pad_context({**latent(), "noise_mask": torch.ones(1)}, 3)
    with pytest.raises(ValueError, match="dictionary"):
        pad_context({}, 3)
    with pytest.raises(ValueError, match="Separate AV"):
        pad_context({"samples": object()}, 3)
    broken = latent()
    broken["samples"][0, 0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        pad_context(broken, 3)


def test_restore_rejects_temporal_changes_mismatched_context_batch_masks_and_nonfinite():
    padded, context = pad_context(latent(), 3)
    for samples in (padded["samples"][:, :, :-5], padded["samples"].repeat(2, 1, 1, 1, 1)):
        with pytest.raises(ValueError, match="preserve batch size and temporal length"):
            restore_context({"samples": samples}, context)
    with pytest.raises(ValueError, match="matching"):
        restore_context(padded, {})
    with pytest.raises(ValueError, match="Masked"):
        restore_context({**padded, "noise_mask": torch.ones(1)}, context)
    padded["samples"][0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        restore_context(padded, context)


def test_registered_nodes_can_be_used_without_comfy_or_upscaler_imports():
    assert NODE_CLASS_MAPPINGS["MiniMaxH3LoopContextPad"] is MiniMaxH3LoopContextPad
    assert NODE_CLASS_MAPPINGS["MiniMaxH3LoopContextRestore"] is MiniMaxH3LoopContextRestore
    padded, context = MiniMaxH3LoopContextPad().pad(latent(), 3)
    out, = MiniMaxH3LoopContextRestore().restore(padded, context)
    assert isinstance(context, LoopContext)
    torch.testing.assert_close(out["samples"], latent()["samples"])
