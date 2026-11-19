"""The numpy resampler matches torchaudio's, and the level helpers behave.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

tests/fixtures/torchaudio_resample.json was made with torchaudio 2.x from the
signal described in it, with reverb_tts.audio.RESAMPLE (Kaiser) or
torchaudio's defaults (Hann).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reverb_addon import audio  # noqa: E402

FIXTURE = json.loads((REPO_ROOT / 'tests' / 'fixtures' / 'torchaudio_resample.json').read_text(encoding='utf-8'))


def _signal() -> np.ndarray:
    t = np.arange(16000 * 2 + 13) / 16000
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * (200 + 1500 * t) * t)).astype(np.float32)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: f"{c['from']}-{c['to']}")
def test_resample_matches_torchaudio(case):
    settings = audio.KAISER_BEST if case['settings'] == 'kaiser' else audio.HANN_DEFAULT
    out = audio.resample(_signal(), case['from'], case['to'], settings)
    assert out.dtype == np.float32 and len(out) == case['length']
    np.testing.assert_allclose(out[::97], case['every_97th'], atol=2e-5)


def test_same_rate_is_unchanged():
    x = _signal()
    assert np.array_equal(audio.resample(x, 24000, 24000), x)


def test_a_quiet_reference_is_raised():
    quiet = (np.sin(np.arange(24000) / 10) * 0.001).astype(np.float32)
    raised = audio.raise_quiet(quiet)
    assert 20 * np.log10(np.sqrt(np.mean(raised ** 2))) == pytest.approx(audio.LEVEL_DB, abs=0.1)
    loud = quiet * 300
    assert audio.raise_quiet(loud) is loud


def test_trim_and_speech_length():
    rate = 24000
    speech = (np.sin(np.arange(rate) / 5) * 0.3).astype(np.float32)
    silence = np.zeros(rate, dtype=np.float32)
    padded = np.concatenate([silence, speech, silence])
    trimmed = audio.trim_silence(padded, rate)
    assert len(trimmed) == pytest.approx(1.2 * rate, abs=rate // 100)
    assert audio.speech_length(padded, rate) == pytest.approx(1.3, abs=0.02)


def test_wav_round_trip(tmp_path):
    x = _signal()[:1000] * 0.9
    audio.write_wav(tmp_path / 'a.wav', x, 16000)
    back, rate = audio.read_wav(tmp_path / 'a.wav')
    assert rate == 16000 and np.abs(back - x).max() < 1e-4
