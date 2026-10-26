import sys
import types

import pytest
import torch

from h3_circular import node
from h3_circular.core import LatentResizer3D


@pytest.fixture
def host(monkeypatch):
    calls = []
    folder = types.ModuleType("folder_paths")
    folder.folder_names_and_paths = {node.MODEL_FOLDER: (["models"], {".safetensors", ".pth"})}
    folder.models_dir = "models"
    folder.get_filename_list = lambda key: ["nested/v1.safetensors", "v1.pth", "notes.txt"]
    folder.get_full_path_or_raise = lambda key, name: "models/" + name
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.ModuleType("comfy.model_management")
    mm = comfy.model_management
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.free_memory = lambda *args: calls.append(("free", args))
    mm.soft_empty_cache = lambda: calls.append(("empty",))
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.utils.load_torch_file = lambda path, safe_load: calls.append(("load", path, safe_load))
    for module in (folder, comfy, comfy.model_management, comfy.utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    model = LatentResizer3D(channels=32, in_blocks=2, out_blocks=2).eval()
    monkeypatch.setattr(node, "load_v1", lambda _: model)
    return calls, model, mm


def test_node_interface_and_execution_with_host_shim(host):
    calls, model, mm = host
    schema = node.MiniMaxH3CircularLatentUpscale.INPUT_TYPES()
    assert schema["required"]["model_name"][0] == ["nested/v1.safetensors", "v1.pth"]
    x = torch.randn(1, 24, 7, 2, 3)
    metadata = [7]
    latent = {"samples": x, "batch_index": metadata}
    out, = node.MiniMaxH3CircularLatentUpscale().upscale(latent, "nested/v1.safetensors", 2., "fp32", "cpu")
    assert out is not latent and out["batch_index"] is metadata
    assert out["samples"].shape == (1, 24, 7, 4, 6)
    assert latent["samples"] is x
    assert calls == [("load", "models/nested/v1.safetensors", True)]
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_no_resize_does_not_load_checkpoint(host):
    calls, _, _ = host
    node.MiniMaxH3CircularLatentUpscale().upscale({"samples": torch.randn(1, 24, 7, 2, 3)},
                                              "nested/v1.safetensors", 1., "fp32", "cpu")
    assert calls == []


def test_mask_fails_before_loading_instead_of_silently_dropping_it(host):
    calls, _, _ = host
    with pytest.raises(ValueError, match="Masked"):
        node.MiniMaxH3CircularLatentUpscale().upscale(
            {"samples": torch.randn(1, 24, 7, 2, 3), "noise_mask": torch.ones(1)},
            "nested/v1.safetensors", 2., "fp32", "cpu")
    assert calls == []


def test_node_cleans_up_and_preserves_cancellation(host):
    _, model, mm = host

    def cancel():
        raise InterruptedError("cancelled")
    mm.throw_exception_if_processing_interrupted = cancel
    with pytest.raises(InterruptedError):
        node.MiniMaxH3CircularLatentUpscale().upscale({"samples": torch.randn(1, 24, 7, 2, 3)},
                                                  "nested/v1.safetensors", 2., "fp32", "cpu")
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_explicit_cuda_device_is_indexed_for_comfy_memory_manager(host, monkeypatch):
    _, model, mm = host
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    moved = []
    monkeypatch.setattr(model, "to", lambda *args, **kwargs: moved.append(kwargs) or model)
    freed = []
    mm.free_memory = lambda size, device: freed.append(device)
    monkeypatch.setattr(node, "upscale_loop", lambda samples, model, scale, dtype, device, interrupt: samples)
    node.MiniMaxH3CircularLatentUpscale().upscale({"samples": torch.randn(1, 24, 7, 2, 3)},
                                              "nested/v1.safetensors", 2., "fp16", "cuda")
    assert freed == [torch.device("cuda:2")]
    assert moved[0]["device"] == torch.device("cuda:2")
    assert moved[-1]["device"] == "cpu"
