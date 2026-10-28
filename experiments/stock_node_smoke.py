"""Run the real stock LBH execute method between Pad and Restore on CPU.

Usage: python experiments/stock_node_smoke.py --source /path/to/LBH/nodes/minimax_h3_latent_upscaler_3d.py --output experiments/stock-node-results.json
Comfy's host/output envelope and weight loading are stubbed in this standalone
process. The stock architecture, normalization, resizing and execute method
run unchanged, with full depth, reduced width and synthetic weights.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch


def import_file(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(813)
    root = Path(__file__).resolve().parents[1]
    import_file("h3_smoke", root / "__init__.py", package=True)
    from h3_smoke.context import pad_context, restore_context
    from h3_smoke.core import LatentResizer3D, with_tail

    folder = types.ModuleType("folder_paths")
    folder.folder_names_and_paths = {"latent_upscale_models": ([], set())}
    sys.modules[folder.__name__] = folder
    # Attention is disabled in the released v1 architecture; fail if it is called.
    einops = types.ModuleType("einops")
    def no_attention(*args, **kwargs):
        raise AssertionError("This v1 smoke test must not exercise attention.")
    einops.rearrange = no_attention
    sys.modules[einops.__name__] = einops
    stock = import_file("lbh_stock_smoke", args.source)
    stock.io.NodeOutput = lambda *values: values
    network = stock.LatentResizer3D(channels=32).eval().requires_grad_(False)
    for layer in network.modules():
        if isinstance(layer, torch.nn.Conv3d):
            torch.nn.init.normal_(layer.weight, std=.018)
            torch.nn.init.normal_(layer.bias, std=.005)
    before = {key: value.clone() for key, value in network.state_dict().items()}
    stock.load_model = lambda name, device, precision: network
    source = {"samples": with_tail(torch.randn(1, 24, 45, 2, 3))}
    input_before = source["samples"].clone()
    padded, context = pad_context(source, 3)
    processed, = stock.MinimaxH3LatentUpscaler3D.execute(
        padded, "synthetic-test-weights", {"mode": stock.UpscaleMode.SCALE_BY, "scale": 2.},
        align=32, enable_temporal_chunking=False, force_unload=True,
        device="cpu", precision="fp32")
    restored = restore_context(processed, context)
    assert restored["samples"].shape == (1, 24, 47, 4, 6)
    assert torch.isfinite(restored["samples"]).all()
    torch.testing.assert_close(source["samples"], input_before, rtol=0, atol=0)
    torch.testing.assert_close(restored["samples"][:, :, -2:], restored["samples"][:, :, :2], rtol=0, atol=0)
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(type(m) is torch.nn.Conv3d for m in network.modules() if isinstance(m, torch.nn.Conv3d))

    # Cross-check the boundary experiment's reduced-width model against the
    # actual stock execute path, including the normalization convention.
    control = LatentResizer3D(channels=32, circular=False).eval()
    control.load_state_dict(before, strict=True)
    mean, std = stock._make_norm_tensors(torch.device("cpu"), torch.float32)
    with torch.inference_mode():
        expected = control((padded["samples"] - mean) / std, 2., (77, 4, 6)) * std + mean
    torch.testing.assert_close(processed["samples"], expected, atol=1e-5, rtol=1e-5)
    result = {"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "stock_execute_ran": True, "host": "shim, not a running ComfyUI server",
              "weights": "synthetic, full depth, 32 channels", "device": "cpu",
              "padded_shape": list(padded["samples"].shape),
              "processed_shape": list(processed["samples"].shape),
              "restored_shape": list(restored["samples"].shape),
              "input_unchanged": True, "weights_unchanged": True, "stock_convs_unchanged": True,
              "canonical_tail": True, "stock_vs_control_max_abs_error": (processed["samples"] - expected).abs().max().item(),
              "gpu_render_tested": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
