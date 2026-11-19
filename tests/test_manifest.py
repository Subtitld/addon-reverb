"""manifest.json, pins.json and the add-on agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reverb_addon import ADDON_ID, __version__, voices  # noqa: E402
from reverb_addon import __main__ as addon  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
PINS = json.loads((REPO_ROOT / 'reverb_addon' / 'pins.json').read_text(encoding='utf-8'))['groups']
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _mb(*groups: str) -> int:
    return round(sum(f['size'] for g in groups for f in PINS[g]['files']) / 1e6)


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True and MANIFEST['compute'] == 'cpu'
    keys = [f['key'] for f in MANIFEST['config_schema']['fields']]
    assert keys == ['threads', 'steps', 'precision', 'voice_ref_audio', 'voice_ref_text']
    [precision] = [f for f in MANIFEST['config_schema']['fields'] if f['key'] == 'precision']
    assert [o['value'] for o in precision['options']] == ['exact', 'fast'] and precision['default'] == 'exact'
    assert 'CC BY-NC 4.0' in MANIFEST['license'] and 'non-commercial' in MANIFEST['summary']


def test_manifest_models_are_the_pins():
    sizes = {'reverb': _mb('core', 'backbone', 'voices'), 'reverb-fast': _mb('backbone-fast'),
             'reverb-codec-encoder': _mb('clone')}
    assert [m['id'] for m in MANIFEST['models']] == list(sizes)
    for model in MANIFEST['models']:
        assert model['size_mb'] == sizes[model['id']], model['id']
        assert model['revision'] == PINS['core']['revision'] and model['bundled'] is False
    assert f"{_mb('core', 'backbone', 'voices')} MB model" in MANIFEST['summary']


def test_pins_are_complete():
    assert set(PINS) == {'core', 'backbone', 'backbone-fast', 'clone', 'voices'}
    release = {'core', 'backbone', 'backbone-fast', 'clone'}
    for name, group in PINS.items():
        if name in release:
            assert group['url'] == f"https://github.com/Subtitld/addon-reverb/releases/download/{group['revision']}"
            assert re.fullmatch(r'models-\d+', group['revision'])
        else:
            assert group['repo'] == 'software-mansion/Reverb-TTS'
            assert re.fullmatch(r'[0-9a-f]{40}', group['revision']), name
        for pin in group['files']:
            assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
    assert len({PINS[g]['revision'] for g in release}) == 1, 'one models release'
    files = {g: {p['path'] for p in PINS[g]['files']} for g in PINS}
    assert files['core'] == {'codec_decoder.onnx', 'tables.npz', 'tokenizer.json', 'MODEL-LICENSE.md'}
    assert files['backbone'] == {'backbone.onnx'} and files['backbone-fast'] == {'backbone_int8.onnx'}
    assert files['clone'] == {'codec_encoder.onnx'}
    assert {f'voices/{v.file}' for v in voices.presets()} | {'README.md'} == files['voices']
    # The release groups land in one folder, as the engine reads them.
    assert len({addon.group_root(g) for g in release}) == 1


@pytest.mark.parametrize('tag,expected', [('pt', 'pt'), ('pt-BR', 'pt'), ('EN_us', 'en'), ('', ''), (None, '')])
def test_language_tags(tag, expected):
    assert voices.split_tag(tag) == expected


def test_the_hosts_rate_is_the_speed():
    assert addon._speed({}) == 1.0
    assert addon._speed({'rate': 50}) == 1.5
    assert addon._speed({'rate': -100}) == 0.5
    assert addon._speed({'rate': 'fast'}) == 1.0


def test_download_groups_follow_the_line():
    assert addon.groups_for(False, 'exact') == ['core', 'backbone', 'voices']
    assert addon.groups_for(True, 'exact') == ['core', 'backbone', 'voices', 'clone']
    assert addon.groups_for(False, 'fast') == ['core', 'backbone-fast', 'voices']


def test_the_model_licence_travels_with_the_files():
    notice = (REPO_ROOT / 'tools' / 'MODEL-LICENSE.md').read_text(encoding='utf-8')
    assert 'CC BY-NC 4.0' in notice and 'non-commercial' in notice and 'Software Mansion' in notice
