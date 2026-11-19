#!/usr/bin/env python3
"""Write reverb_addon/pins.json: every file the add-on downloads, by size and sha256.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python tools/pin_models.py MODELS_DIR

Files are grouped by what needs them, and a group downloads the first time a
line needs it:

- core: the codec decoder, the tables, the tokenizer and the model files'
  licence. They are tools/export_models.py's output in MODELS_DIR, pinned
  from there, and uploaded as they are to this repository's MODELS_TAG
  release;
- backbone: the backbone at full precision (the default);
- backbone-fast: the backbone in dynamic int8 (Configure's "Fast");
- clone: the codec encoder, for the speaker's own voice (same release);
- voices: the 16 preset voices and the model card, from Reverb's own
  repository at v1.0.0.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_TAG = 'models-1'
RELEASE_URL = f'https://github.com/Subtitld/addon-reverb/releases/download/{MODELS_TAG}'
REPO = 'software-mansion/Reverb-TTS'
REVISION = '72be799c8fb6d56a206d260b425c48bf3bd5943d'  # the v1.0.0 tag

RELEASE_GROUPS = {
    'core': ['codec_decoder.onnx', 'tables.npz', 'tokenizer.json', 'MODEL-LICENSE.md'],
    'backbone': ['backbone.onnx'],
    'backbone-fast': ['backbone_int8.onnx'],
    'clone': ['codec_encoder.onnx'],
}
HF_GROUPS = {'voices': ['voices/', 'README.md']}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    models = Path(argv[0])
    groups = {}
    for name, files in RELEASE_GROUPS.items():
        pins = [{'path': f, 'size': (models / f).stat().st_size, 'sha256': sha256(models / f)} for f in files]
        groups[name] = {'url': RELEASE_URL, 'revision': MODELS_TAG, 'files': pins}
    items = json.loads(get(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=1'))
    tree = {item['path']: item for item in items if item.get('type') == 'file'}
    for name, wanted in HF_GROUPS.items():
        paths = sorted(p for p in tree if any(p == w or (w.endswith('/') and p.startswith(w)) for w in wanted))
        pins = []
        for path in paths:
            meta = tree[path]
            sha = (meta.get('lfs') or {}).get('oid')
            if not sha:
                data = get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{path}')
                if len(data) != meta['size']:
                    raise SystemExit(f'{path}: got {len(data)} bytes, the tree says {meta["size"]}')
                sha = hashlib.sha256(data).hexdigest()
            pins.append({'path': path, 'size': meta['size'], 'sha256': sha})
        groups[name] = {'repo': REPO, 'revision': REVISION, 'files': pins}
    for name, group in groups.items():
        print(f"{name}: {len(group['files'])} files, {sum(f['size'] for f in group['files']) / 1e6:,.0f} MB")
    out = ROOT / 'reverb_addon' / 'pins.json'
    out.write_text(json.dumps({'groups': groups}, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
