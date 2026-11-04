"""Opt-in integration check against a real ComfyUI checkout, on CPU.

Run with a Python environment containing torch, pytest and ComfyUI's dependencies:
    python tests/check_native_comfy.py --comfyui-root /path/to/ComfyUI
This never starts a server or downloads/loads production model weights.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui_root.resolve()))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    from comfy import ops
    from comfy.ldm.minimax import latent_upscaler as native

    package = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("h3_circular", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from h3_circular.core import LatentResizer3D, upscale_loop, with_tail
    from h3_circular.native_backend import circular_forward, validate_model

    torch.set_num_threads(2)
    torch.manual_seed(401)
    reference = LatentResizer3D(channels=32).eval()
    for layer in reference.modules():
        if isinstance(layer, torch.nn.Conv3d):
            torch.nn.init.normal_(layer.weight, std=.018)
            torch.nn.init.normal_(layer.bias, std=.005)
    model = native.MiniMaxH3LatentUpscaler(channels=32, dtype=torch.float32,
                                         device="cpu", operations=ops.disable_weight_init).eval()
    model.load_state_dict(reference.state_dict())
    validate_model(model, native)
    report = {
        "comfyui_commit": subprocess.check_output(["git", "-C", str(args.comfyui_root),
                                                   "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
        "device": "cpu", "channels": 32, "residual_blocks_per_stage": 12,
        "weights": "seeded synthetic; no released checkpoint rendered",
        "checks": [],
    }
    with torch.inference_mode():
        for length, batch in ((5, 1), (20, 2), (45, 2)):
            x = torch.randn(batch, 24, length, 2, 3)
            before = x.clone()
            state = {key: value.clone() for key, value in model.state_dict().items()}
            stock_before = model(x, (length, 4, 6))
            actual = circular_forward(model, x, (length, 4, 6), native, ops)
            expected = upscale_loop(with_tail(x), reference, 2., torch.float32,
                                     torch.device("cpu"))[:, :, :-2]
            torch.testing.assert_close(actual, expected, atol=8e-5, rtol=8e-5)
            rotation_error = 0.
            for shift in (1, 5, 16):
                rotated = circular_forward(model, x.roll(shift, 2), (length, 4, 6), native, ops)
                torch.testing.assert_close(rotated, actual.roll(shift, 2), atol=8e-5, rtol=8e-5)
                rotation_error = max(rotation_error, float((rotated - actual.roll(shift, 2)).abs().max()))
            torch.testing.assert_close(x, before, atol=0, rtol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], atol=0, rtol=0)
            torch.testing.assert_close(model(x, (length, 4, 6)), stock_before, atol=0, rtol=0)
            report["checks"].append({"body_length": length, "batch": batch,
                                      "max_reference_error": float((actual - expected).abs().max()),
                                      "max_rotation_error": rotation_error,
                                      "input_weights_and_stock_output_unchanged": True})
    report["passed"] = True
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
