"""Check schedule selection and preservation of the working generation graph."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'workflows/LOOP-v1.1-Regenerate-3step.json').read_text())
GRAPH = json.loads((ROOT / 'workflows/LOOP-v1.2-Regenerate-Sigma-Selector.json').read_text())
REFINER = 3660768273029208


def nodes(graph):
    return {n['id']: n for n in graph['nodes']}


def edges(graph):
    index = nodes(graph)
    return {(l[1], l[2], l[3], index[l[3]]['inputs'][l[4]]['name']) for l in graph['links']}


def test_existing_generation_settings_and_edges_are_preserved():
    before, after = nodes(BASE), nodes(GRAPH)
    assert len(after) - len(before) == 4
    assert GRAPH['definitions'] == BASE['definitions']
    for key, old in before.items():
        new = after[key]
        for field in ('type', 'mode', 'flags'):
            assert new.get(field) == old.get(field)
        if old['type'] not in ('Note', 'MarkdownNote'):
            assert 'title' not in new
            if key != 4353752495738589:
                assert new.get('widgets_values') == old.get('widgets_values'), key
    unchanged = {e for e in edges(BASE) if e[2:] != (REFINER, 'sigmas')}
    retained = {e for e in edges(GRAPH) if e[0] in before and e[2] in before}
    assert retained == unchanged
    assert sum(n['type'] == 'ModelPreviewOverrideKJ' for n in after.values()) == 1


def test_dropdown_routes_each_existing_preset_to_only_the_refiner():
    index = nodes(GRAPH)
    revision = GRAPH['extra']['sigma_selector_revision']
    selector = index[int(revision['aliases']['sigma_selector'])]
    assert selector['type'] == 'BranchNode'
    labels = revision['labels']
    assert selector['widgets_values'] == [labels[0], labels]
    ports = [p for p in selector['inputs'] if p['name'].startswith('autogrow.') and p['link']]
    assert [p['label'] for p in ports] == labels
    assert [p['name'] for p in ports] == [f'autogrow.branch{i}' for i in range(4)]
    links = {l[0]: l for l in GRAPH['links']}
    sigma_id = int(BASE['extra']['regenerate_revision']['aliases']['sigmas'])
    for port, preset in zip(ports, revision['presets']):
        upstream = index[links[port['link']][1]]
        reference = json.loads((ROOT / f'workflows/LOOP-v1.1-Regenerate-{preset}.json').read_text())
        assert upstream['type'] == 'ManualSigmas'
        assert upstream['widgets_values'] == nodes(reference)[sigma_id]['widgets_values']
    assert {e for e in edges(GRAPH) if e[0] == selector['id']} == {
        (selector['id'], 0, REFINER, 'sigmas')
    }


def test_canvas_nodes_do_not_overlap_and_stages_contain_their_nodes():
    boxes = []
    for node in GRAPH['nodes']:
        x, y = node['pos']
        w, h = node['size']
        # Include native node title height in the collision check.
        box = (x, y - 30, x + w, y + h)
        for other, previous in boxes:
            assert not (box[0] < previous[2] and previous[0] < box[2]
                        and box[1] < previous[3] and previous[1] < box[3]), (node['id'], other)
        boxes.append((node['id'], box))
        if node['id'] in (4416792996266400, 4416792996266401, 4416792996266402):
            continue
        groups = [g['bounding'] for g in GRAPH['groups']]
        assert sum(gx <= x and gy <= y - 30 and gx + gw >= x + w and gy + gh >= y + h
                   for gx, gy, gw, gh in groups) == 1, node['id']
