"""CPU integration checks of pinned upstream wrapper/node code, without H3 weights.

Run with PyTorch installed. --sources-dir reuses previously inspected sources;
otherwise downloads only small Python sources into a temporary directory.
The transformer and VAE are synthetic test doubles, not a GPU quality test.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS
import tempfile
import urllib.request

import torch

SOURCES = {
    "loop-modal-pin.py": "https://raw.githubusercontent.com/pxl-pshr/ComfyUI-MiniMaxH3-Loop/733647e/__init__.py",
    "comfy-nodes-audio.py": "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/426f5ee/comfy_extras/nodes_audio.py",
    "comfy-nodes-ltxv.py": "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/426f5ee/comfy_extras/nodes_lt.py",
    "comfy-sigmas.py": "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/426f5ee/comfy_extras/nodes_custom_sampler.py",
}


def load_defs(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class Pair:
    is_nested = True

    def __init__(self, streams):
        self.streams = tuple(streams)

    def unbind(self):
        return self.streams


def unpack(flat, shapes):
    sizes = [torch.tensor(s[1:]).prod().item() for s in shapes]
    return [part.reshape(shape) for part, shape in zip(flat.split(sizes, dim=-1), shapes)]


class Model:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}

    def clone(self):
        return copy.deepcopy(self)

    def get_all_wrappers(self, kind):
        return [value for (key, _), value in self.wrappers.items() if key == kind]

    def add_wrapper_with_key(self, kind, key, value):
        self.wrappers[kind, key] = value


def main(directory):
    io = NS(ComfyNode=object, NodeOutput=lambda *values: values)
    comfy = NS(
        utils=NS(unpack_latents=unpack),
        nested_tensor=NS(NestedTensor=Pair),
        patcher_extension=NS(WrappersMP=NS(DIFFUSION_MODEL="model", SAMPLER_SAMPLE="sampler"),
                             get_all_wrappers=lambda *args, **kwargs: []),
    )
    env = dict(torch=torch, comfy=comfy, io=io, IO=io, re=re,
               CLIP_LATENTS=5, VIDEO_TAIL=2, AUDIO_PER_CLIP=17 * 40 / 24,
               MiniMaxH3FunControlPatch=type("FunControl", (), {}))
    load_defs(directory / "loop-modal-pin.py", ["audio_span", "wrap_pad", "joint_weight", "loop_wrapper",
              "has_fun_controlnet", "loop_sampler_wrapper", "MiniMaxH3Loop", "MiniMaxH3LoopDecode"], env)
    load_defs(directory / "comfy-nodes-audio.py", ["EmptyAudio", "VAEEncodeAudio"], env)
    load_defs(directory / "comfy-nodes-ltxv.py", ["LTXVConcatAVLatent"], env)
    load_defs(directory / "comfy-sigmas.py", ["ManualSigmas"], env)

    original = Model()
    model, = env["MiniMaxH3Loop"]().patch(original, pad_clips=2, views=1)
    assert original.model_options == {"transformer_options": {}} and original.wrappers == {}
    assert set(model.wrappers) == {("model", "minimax_h3_loop"), ("sampler", "minimax_h3_loop")}

    # The zero waveform must go through the VAE; the encoded silence need not be zero.
    class AudioVAE:
        audio_sample_rate = 32000

        def encode(self, waveform):
            assert waveform.shape[2] == 2 and torch.count_nonzero(waveform) == 0
            samples = waveform.shape[1]
            assert samples % 800 == 0
            return torch.full((1, 32, 2, samples // 800), 0.125)

    durations = []
    for clips in (1, 6, 9, 12, 21):
        audio_t = round((clips * 17 + 5) * 40 / 24)
        audio, = env["EmptyAudio"].execute(audio_t / 40, 32000, 2)
        encoded, = env["VAEEncodeAudio"].execute(AudioVAE(), audio)
        assert encoded["samples"].shape[-1] == audio_t
        durations.append(audio_t)
    audio, = env["EmptyAudio"].execute(263 / 40, 32000, 2)
    encoded, = env["VAEEncodeAudio"].execute(AudioVAE(), audio)
    encoded["noise_mask"] = torch.zeros(1, 1, 1, 1)  # stock SetLatentNoiseMask output

    generator = torch.Generator().manual_seed(42)
    body = torch.randn((1, 24, 45, 2, 2), generator=generator)
    clean = torch.cat((body, body[:, :, :2]), dim=2)
    av, = env["LTXVConcatAVLatent"].execute({"samples": clean}, encoded)
    video_mask, audio_mask = av["noise_mask"].unbind()
    assert torch.all(video_mask == 1) and torch.count_nonzero(audio_mask) == 0
    audio_clean = encoded["samples"]
    shapes = [tuple(clean.shape), tuple(audio_clean.shape)]
    video_noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(43))
    packed_noise = torch.cat((video_noise.flatten(1), torch.zeros_like(audio_clean).flatten(1)), dim=-1)
    guider = NS(inner_model=NS(latent_shapes=shapes))
    observed = []
    schedules = [[0.909375, 6 / 7, 2 / 3, 0], [0.7, 0.4941176470588235, 0]]

    # Independent index construction, so checks don't use wrap_pad as their oracle.
    vi = torch.arange(-10, 57) % 45
    ai = torch.arange(-57, 320) % 255
    for schedule in schedules:
        sigmas, = env["ManualSigmas"].execute(", ".join(str(s) for s in schedule))
        expected_fresh = torch.randn(clean.shape, generator=torch.Generator().manual_seed(44))
        expected_swap = (expected_fresh - video_noise)[:, :, :45][:, :, vi].clone()
        expected_swap[:, :, 10:55] = 0
        options = model.model_options["transformer_options"]
        steps = []

        def sampling_executor(guider, sigmas, extra_args, callback, noise):
            x = (1 - sigmas[0]) * clean + sigmas[0] * video_noise
            start = x.clone()
            for sigma, following in zip(sigmas[:-1], sigmas[1:]):
                options["sigmas"] = sigma.reshape(1)

                def transformer(streams, timestep, context, transformer_options, **kwargs):
                    v, a = streams
                    expected = x[:, :, :45][:, :, vi] + sigma * expected_swap
                    torch.testing.assert_close(v, expected)
                    torch.testing.assert_close(a, audio_clean[..., :255][..., ai])
                    assert torch.count_nonzero(kwargs["audio_denoise_mask"]) == 0
                    assert v.shape[2] == 67 and a.shape[-1] == 377
                    # A small temporal filter makes seam neighbors affect the prediction.
                    pred = torch.nn.functional.avg_pool3d(v, (3, 1, 1), stride=1, padding=(1, 0, 0))
                    steps.append({"sigma": float(sigma), "video_context_T": v.shape[2], "audio_context_T": a.shape[-1]})
                    return [pred, torch.zeros_like(a)]

                v, a = model.wrappers["model", "minimax_h3_loop"](
                    transformer, [x, audio_clean], sigma.reshape(1) * 1000, None, options,
                    audio_denoise_mask=torch.zeros(1, 1, 2, 263))
                assert v.shape == x.shape and a.shape == audio_clean.shape
                torch.testing.assert_close(v[:, :, -2:], v[:, :, :2], rtol=0, atol=0)
                # Comfy's CONST representation: x0 = x - sigma * model_output.
                # Euler therefore advances x by (following - sigma) * model_output.
                x = x + (following - sigma) * v
            assert not torch.equal(x[:, :, :45], start[:, :, :45])
            assert torch.isfinite(x).all()
            return x

        result = model.wrappers["sampler", "minimax_h3_loop"](
            sampling_executor, guider, sigmas, {"model_options": model.model_options, "seed": 43}, None, packed_noise)
        assert len(steps) == len(schedule) - 1
        observed.append(steps)

    # Check that the original bookkeeping tail never reaches the visible decode.
    decode_seen = []

    class VideoVAE:
        def decode(self, latent):
            decode_seen.append(latent.clone())
            frames = (latent.shape[2] - 2) // 5 * 17 + 5
            return torch.arange(frames).float().reshape(frames, 1, 1, 1).expand(-1, 2, 2, 3)

    env["vae_decode_audio"] = lambda vae, samples: {
        "waveform": torch.zeros(1, 2, samples["samples"].shape[-1] * 800), "sample_rate": 32000,
    }
    # Deliberately poison the unused tail: Loop Decode must construct its own context.
    result[:, :, -2:] = 1e6
    images, audio = env["MiniMaxH3LoopDecode"]().decode(
        {"samples": Pair((result, audio_clean))}, VideoVAE(), AudioVAE(), 1)
    torch.testing.assert_close(decode_seen[0], result[:, :, :45][:, :, torch.arange(-5, 52) % 45])
    assert images.shape[0] == 153 and images[0, 0, 0, 0] == 17
    assert audio["waveform"].shape[-1] == 204000

    return {
        "passed": True, "gpu_render_tested": False,
        "test_doubles": ["transformer temporal filter", "audio/video VAEs", "Comfy host facade"],
        "checks": ["model clone isolation", "encoded nonzero silence with zero noise mask",
                   "automatic audio duration at five loop lengths", "fresh padding noise at each sigma",
                   "end/start context on every 3-step and 2-step model call",
                   "predicted tail rebuild", "Loop Decode ignores poisoned tail and returns 153 frames"],
        "sampling_calls": observed, "tested_audio_lengths": durations,
        "sources": {name: {"url": url, "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
                    for name, url in SOURCES.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="h3-loop-check-") as tmp:
        directory = args.sources_dir or Path(tmp)
        if args.sources_dir is None:
            for name, url in SOURCES.items():
                with urllib.request.urlopen(url, timeout=30) as response:
                    (directory / name).write_bytes(response.read())
        report = main(directory)
        text = json.dumps(report, indent=2) + "\n"
        if args.out:
            args.out.write_text(text)
        print(text)
