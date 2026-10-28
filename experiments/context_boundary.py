"""Compare external pad/crop with circular convolutions; no model download.

Run: python experiments/context_boundary.py --output experiments/context-results.json
This uses full-depth LBH-v1 structure with 32 channels and synthetic weights.
It measures boundary behavior, not trained-model picture quality.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_context_experiment", PACKAGE / "__init__.py",
                                            submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from h3_context_experiment.core import LatentResizer3D, with_tail
from h3_context_experiment.context import pad_context, restore_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(813)
    circular = LatentResizer3D(channels=32).eval()
    for layer in circular.modules():
        if isinstance(layer, torch.nn.Conv3d):
            torch.nn.init.normal_(layer.weight, std=.018)
            torch.nn.init.normal_(layer.bias, std=.005)
    stock = LatentResizer3D(channels=32, circular=False).eval()
    stock.load_state_dict(circular.state_dict())
    body = torch.randn(1, 24, 45, 2, 3)

    def process(x, pad_clips):
        if pad_clips:
            padded, context = pad_context({"samples": with_tail(x)}, pad_clips)
            samples = padded["samples"]
        else:
            samples = with_tail(x)
        y = stock(samples, 2., (samples.shape[2], 4, 6))
        if pad_clips:
            y = restore_context({"samples": y}, context)["samples"]
        return y[:, :, :-2]

    rows = []
    with torch.inference_mode():
        expected = circular(body, 2., (45, 4, 6))
        shifted = circular(body.roll(5, 2), 2., (45, 4, 6)).roll(-5, 2)
        circular_rotation_max = (expected - shifted).abs().max().item()
        for pad_clips in (0, 1, 3, 15, 27, 45):
            result = process(body, pad_clips)
            shifted = process(body.roll(5, 2), pad_clips).roll(-5, 2)
            rows.append({
                "pad_clips": pad_clips,
                "input_latents": 47 + 10 * pad_clips,
                "rotation_max_abs_error": (result - shifted).abs().max().item(),
                "relative_rmse_vs_circular": (
                    (result - expected).square().mean().sqrt()
                    / expected.square().mean().sqrt()).item(),
            })
    result = {
        "torch": torch.__version__, "device": "cpu", "dtype": "float32",
        "seed": 813, "body_latents": 45, "spatial_scale": 2,
        "channels_tested": 32, "production_channels": 512,
        "weights": "synthetic; identical across circular and stock controls",
        "temporal_chunking": False,
        "circular_rotation_max_abs_error": circular_rotation_max,
        "rows": rows,
        "interpretation": "External context reduces but does not eliminate origin dependence. "
                          "Do not infer trained-model quality or Regenerate compatibility from these numbers.",
        "gpu_render_tested": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
