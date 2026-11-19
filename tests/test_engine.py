"""Real synthesis with the pinned model, from source.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Every download group is fetched into $SUBTITLD_MODELS_DIR, or
~/.cache/subtitld/models, on the first run (1.3 GB), checked against
pins.json.

tests/fixtures/package_log_probs.json holds what Reverb's own package (torch,
int8 weights expanded to fp32) predicts for codebooks 0-1 of one partly
filled take; the exact backbone has to predict the same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reverb_addon import __main__ as addon, download, engine, voices  # noqa: E402

pytestmark = pytest.mark.model
FIXTURE = json.loads((REPO_ROOT / 'tests' / 'fixtures' / 'package_log_probs.json').read_text(encoding='utf-8'))
LINES = {
    'en': 'We have to leave before sunrise.',
    'de': 'Wir müssen vor Sonnenaufgang aufbrechen.',
    'fr': 'Nous devons partir avant le lever du soleil.',
    'es': 'Tenemos que salir antes del amanecer.',
    'it': "Dobbiamo partire prima dell'alba.",
    'pl': 'Musimy wyjść przed wschodem słońca.',
    'pt': 'Temos de sair antes do nascer do sol.',
}


def _load(precision: str) -> engine.Reverb:
    group, name = addon._BACKBONES[precision]
    for g in ('core', group, 'voices', 'clone'):
        download.ensure(addon.pins(g), lambda *_: None, lambda: False)
    return engine.Reverb(addon.group_root('core'), addon.group_root(group) / name, threads=engine.default_threads(),
                         encoder_path=addon.group_root('clone') / 'codec_encoder.onnx')


@pytest.fixture(scope='module')
def model():
    return _load('exact')


def _preset(name: str) -> engine.Voice:
    return engine.load_preset(addon.group_root('voices') / 'voices' / name)


def _log_probs(model: engine.Reverb) -> np.ndarray:
    """Codebooks 0-1 of the fixture's take, as the sampler's first kind of pass sees them."""
    voice = _preset(FIXTURE['voice'])
    ids = np.asarray(model.tokenizer.encode(engine._prompt(voice.text, FIXTURE['text'], 'en')).ids)
    prefix = np.concatenate([model.text_table[ids] @ model.text_project.T, model._embed_audio(voice.codes)])
    P, F = prefix.shape[0], FIXTURE['frames']
    empty = np.zeros((engine.LAYERS, 1, engine.KV_HEADS, 0, engine.HEAD_DIM), np.float32)
    _, keys, values = model.backbone.run(None, {
        'embeds': prefix[None].astype(np.float32), 'mask': np.zeros((1, 1, P, P), np.float32),
        'positions': np.arange(P, dtype=np.int64)[None], 'past_keys': empty, 'past_values': empty})
    tokens = np.asarray(FIXTURE['tokens'], dtype=np.int64)
    hidden = model.backbone.run(['hidden'], {
        'embeds': model._embed_audio(tokens)[None].astype(np.float32), 'mask': np.zeros((1, 1, F, P + F), np.float32),
        'positions': (P + np.arange(F, dtype=np.int64))[None], 'past_keys': keys, 'past_values': values})[0][0]
    logits = (hidden @ model.audio_heads[: 2 * engine.CODES].T).reshape(F, 2, engine.CODES).transpose(1, 0, 2)
    return engine._log_softmax(logits)[..., : engine.MASK]


def test_the_exact_backbone_predicts_what_the_package_does(model):
    lp = _log_probs(model)
    assert (lp.argmax(-1) == np.asarray(FIXTURE['argmax'])).mean() >= 0.97
    np.testing.assert_allclose(lp.max(-1), FIXTURE['max_log_prob'], atol=0.05)


def test_the_fast_backbone_runs():
    fast = _load('fast')
    wav = fast.synthesize(LINES['en'], _preset('en_male_narrator_1.json'), 'en', seed=1)
    assert 1.0 < len(wav) / engine.SAMPLE_RATE < 5


@pytest.mark.parametrize('language', sorted(LINES))
def test_speaks_each_language(model, language):
    voice = next(v for v in voices.presets() if v.file.startswith(language))
    wav = model.synthesize(LINES[language], _preset(voice.file), language, seed=3)
    seconds = len(wav) / engine.SAMPLE_RATE
    assert 1.0 < seconds < 6, seconds
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
    assert 0.01 < float(np.sqrt(np.mean(wav ** 2))) < 0.5


def test_a_seed_gives_the_same_line(model):
    voice = _preset('pt_female_standard.json')
    one = model.synthesize(LINES['pt'], voice, 'pt', seed=11)
    again = model.synthesize(LINES['pt'], voice, 'pt', seed=11)
    other = model.synthesize(LINES['pt'], voice, 'pt', seed=12)
    assert np.array_equal(one, again)
    assert len(one) != len(other) or not np.array_equal(one, other)


def test_clones_a_clip_with_its_transcript(model):
    said = 'Long before the first cities rose, people gathered around fires to tell stories.'
    clip = model.synthesize(said, _preset('en_male_narrator_1.json'), 'en', seed=2)
    voice = model.clone_voice(clip, engine.SAMPLE_RATE, said, 'en')
    assert voice.codes.shape[0] == engine.CODEBOOKS
    assert abs(voice.codes.shape[1] - len(clip) / engine.HOP) <= 1
    assert 0 < voice.pace_frames <= voice.codes.shape[1]
    wav = model.synthesize(LINES['it'], voice, 'it', seed=2)
    assert 1.0 < len(wav) / engine.SAMPLE_RATE < 6


def test_long_text_comes_in_pieces_and_reports_progress(model):
    voice = _preset('en_female_narrator.json')
    line = ' '.join(['The river ran quietly past the old mill, and nobody noticed the boat.'] * 8)
    assert len(model.plan(line, voice, 'en')) > 1
    seen = []
    wav = model.synthesize(line, voice, 'en', seed=1, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] > 0.9
    assert len(wav) / engine.SAMPLE_RATE > 20


def test_a_cancel_stops_between_passes(model):
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(engine.Cancelled):
        model.synthesize(LINES['en'], _preset('en_male_angry.json'), 'en', cancelled=cancelled)
