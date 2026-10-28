"""Protect the user's attached graph and keep sigma comparisons controlled."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / 'workflows/sources/LOOP-v1.0-8step_lora.json').read_text())
PRESETS = ('3step', '2step-gentler', '3step-start080', '4step-shift6')
REMOVED = 3313622723251107
NOTES = {4416792996266400, 4416792996266401, 4416792996266402}
SAVE = 4353752495738589


def load(preset):
    return json.loads((ROOT / 'workflows' / f'LOOP-v1.1-Regenerate-{preset}.json').read_text())


def node_map(graph):
    return {n['id']: n for n in graph['nodes']}


def edges(graph):
    nodes = node_map(graph)
    return {(link[1], link[2], link[3], nodes[link[3]]['inputs'][link[4]]['name'], link[5])
            for link in graph['links']}


@pytest.mark.parametrize('preset', PRESETS)
def test_user_layout_settings_subgraphs_and_bypass_modes_preserved(preset):
    graph = load(preset)
    nodes = node_map(graph)
    assert graph['definitions'] == SOURCE['definitions']
    assert set(node_map(SOURCE)) - set(nodes) == {REMOVED}
    for old in SOURCE['nodes']:
        if old['id'] == REMOVED:
            continue
        new = nodes[old['id']]
        for key in ('pos', 'size', 'mode', 'flags', 'type'):
            assert new.get(key) == old.get(key), (old['id'], key)
        if old['id'] not in NOTES | {SAVE}:
            assert new.get('widgets_values') == old.get('widgets_values'), old['id']
        if new['type'] not in ('Note', 'MarkdownNote'):
            assert 'title' not in new
    assert sum(n['type'] == 'ModelPreviewOverrideKJ' for n in nodes.values()) == 1


def test_all_unrelated_edges_preserved():
    graph = load('3step')
    allowed_destinations = {
        (3660768273029208, 'guider'), (3660768273029208, 'sigmas'),
        (3891363586297582, 'audio_latent'), (1736980650663141, 'audio'),
        (4416792996266396, 'model.audio'),
    }
    old = {e for e in edges(SOURCE) if REMOVED not in (e[0], e[2])
           and (e[2], e[3]) not in allowed_destinations}
    kept_ids = set(node_map(SOURCE)) - {REMOVED}
    new = {e for e in edges(graph) if e[0] in kept_ids and e[2] in kept_ids
           and (e[2], e[3]) not in allowed_destinations}
    assert new == old


def test_refine_model_bypasses_preview_and_final_audio_comes_from_draft():
    graph = load('3step')
    nodes = node_map(graph)
    aliases = {k: int(v) for k, v in graph['extra']['regenerate_revision']['aliases'].items()}

    def incoming(n, socket):
        return next(e[0] for e in edges(graph) if e[2] == n and e[3] == socket)

    assert incoming(3660768273029208, 'guider') == aliases['guider']
    assert incoming(aliases['guider'], 'model') == aliases['loop']
    assert incoming(aliases['loop'], 'model') == aliases['loras']
    assert incoming(aliases['loras'], 'model') == aliases['base']
    assert nodes[aliases['base']]['widgets_values'][0] == 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'
    assert nodes[aliases['loras']]['widgets_values'] == [
        2, 'minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors', 1.0, True,
        'minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors', 1.0, True,
    ]
    assert incoming(aliases['text'], 'text') == 4416792996266410
    assert incoming(3660768273029208, 'noise') == 2625031668252728
    assert nodes[2625031668252728]['widgets_values'] == [111, 'fixed']
    assert incoming(1736980650663141, 'audio') == 4416792996266380
    assert incoming(4416792996266396, 'model.audio') == 4416792996266380
    assert incoming(1736980650663141, 'images') == 3902486688621038
    assert incoming(3891363586297582, 'audio_latent') == aliases['freeze']
    assert nodes[aliases['zero']]['widgets_values'] == [0.0, 1, 1]


def test_variants_change_only_sigmas_output_prefix_and_preset_metadata():
    base = load('3step')
    sigma_id = int(base['extra']['regenerate_revision']['aliases']['sigmas'])
    for preset in PRESETS[1:]:
        other = load(preset)
        before, after = node_map(base), node_map(other)
        assert set(before) == set(after)
        assert base['links'] == other['links']
        assert base['definitions'] == other['definitions']
        for key in before:
            if key not in (sigma_id, SAVE):
                # Comfy CLI may normalize node order when replaying an edit.
                a, b = dict(before[key]), dict(after[key])
                a.pop('order', None)
                b.pop('order', None)
                assert a == b, key
        sigma_text = after[sigma_id]['widgets_values'][0]
        sigmas = [float(x) for x in sigma_text.split(',')]
        assert sigmas[-1] == 0 and all(a > b for a, b in zip(sigmas, sigmas[1:]))
        assert len(sigmas) - 1 == int(preset[0])
