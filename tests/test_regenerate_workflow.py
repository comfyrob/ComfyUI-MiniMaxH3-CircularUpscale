"""Graph contracts that prevent accidentally refining with the draft model."""
import json
from pathlib import Path

GRAPH = json.loads((Path(__file__).resolve().parents[1] / "workflows" /
                    "MiniMax-H3-Regenerate-Loop-Refinement-3Step.json").read_text())
NODES = {n["id"]: n for n in GRAPH["nodes"]}
LINKS = {link[0]: link for link in GRAPH["links"]}


def only(kind):
    nodes = [n for n in NODES.values() if n["type"] == kind]
    assert len(nodes) == 1
    return nodes[0]


def source(node, socket):
    link_id = next(x["link"] for x in node["inputs"] if x["name"] == socket)
    return NODES[LINKS[link_id][1]]


def refinement():
    return source(only("MiniMaxH3LoopDecode"), "samples")


def test_refiner_model_branch_is_loop_patched_and_clean():
    guider = source(refinement(), "guider")
    assert guider["type"] == "BasicGuider"
    preview = source(guider, "model")
    assert preview["type"] == "ModelPreviewOverrideKJ"
    loop = source(preview, "model")
    assert loop["type"] == "MiniMaxH3Loop" and loop["widgets_values"] == [2, 1]
    refiner = source(loop, "model")
    assert refiner["widgets_values"] == ["minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors", 1.0]
    turbo = source(refiner, "model")
    assert turbo["widgets_values"] == ["minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", 1.0]
    base = source(turbo, "model")
    assert base["type"] == "UNETLoader"
    assert base["widgets_values"][0] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert source(guider, "conditioning")["type"] == "CLIPTextEncode"


def test_refine_uses_published_sigmas_euler_and_fresh_noise():
    sample = refinement()
    assert sample["type"] == "SamplerCustomAdvanced"
    sigma = source(sample, "sigmas")
    assert sigma["type"] == "ManualSigmas"
    assert [float(x) for x in sigma["widgets_values"][0].split(",")] == [0.909375, 6 / 7, 2 / 3, 0]
    assert source(sample, "sampler")["widgets_values"] == ["euler"]
    assert source(sample, "noise")["type"] == "RandomNoise"
    assert source(sample, "noise")["widgets_values"][0] == 43


def test_refine_receives_circular_upscale_and_frozen_encoded_silence():
    concat = source(refinement(), "latent_image")
    assert concat["type"] == "LTXVConcatAVLatent"
    upscale = source(concat, "video_latent")
    assert upscale["type"] == "MiniMaxH3CircularLatentUpscale"
    assert upscale["widgets_values"] == ["minimax_h3_latent_upscaler_3d_bf16.safetensors", 2.0, "fp16", "cuda"]
    frozen = source(concat, "audio_latent")
    assert frozen["type"] == "SetLatentNoiseMask"
    assert source(frozen, "mask")["widgets_values"] == [0.0, 1, 1]
    encode = source(frozen, "samples")
    assert encode["type"] == "VAEEncodeAudio"
    assert "audio_vae" in source(encode, "vae")["widgets_values"][0]
    silence = source(encode, "audio")
    assert silence["type"] == "EmptyAudio"
    seconds = source(silence, "duration")
    assert seconds["widgets_values"] == ["a / 40"]
    assert source(seconds, "a")["type"] == "GetAudioLatentSize"


def test_no_generic_context_adapters_and_prompt_is_shared():
    assert not any("LoopContext" in n["type"] for n in NODES.values())
    prompt = source(only("MiniMaxH3ReferenceToVideo"), "prompt")
    assert prompt["type"] == "PrimitiveStringMultiline"
    assert source(only("CLIPTextEncode"), "text")["id"] == prompt["id"]
    assert not any("lms_v1.0" in str(n.get("widgets_values", []))
                   for n in NODES.values() if n["type"] == "LoraLoaderModelOnly")
