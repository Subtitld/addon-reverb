"""Text preparation, length estimates and the transcript's language guess.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reverb_addon import text  # noqa: E402


def test_numbers_are_spelled_out():
    assert text.prepare('I have 3 cats (and a dog).', 'en') == 'I have three cats and a dog.'
    assert 'doze' in text.prepare('Custa 12 reais.', 'pt')


@pytest.mark.parametrize('sentence,language', [
    ('Look Celia, we have to follow our passions; you have your robotics.', 'en'),
    ('Ich habe die Katze nicht gesehen, und du?', 'de'),
    ('No sé lo que quieres de mí, pero no voy a esperar.', 'es'),
    ('Je ne sais pas ce que tu veux, mais je pars.', 'fr'),
    ('Non so che cosa vuoi, ma io non aspetto.', 'it'),
    ('Nie wiem, czego chcesz, ale nie będę czekać.', 'pl'),
    ('Não sei o que você quer, mas eu não vou esperar.', 'pt'),
])
def test_guess_language(sentence, language):
    assert text.guess_language(sentence, fallback='en' if language != 'en' else 'pt') == language


def test_guess_falls_back_when_nothing_tells():
    assert text.guess_language('Okay!', 'it') == 'it'
    assert text.guess_language('', 'pl') == 'pl'


def test_length_follows_the_speakers_pace():
    slow, fast = text.pace('a' * 100, 400), text.pace('a' * 100, 100)
    line = 'This sentence is long enough to be measured properly by the estimate.'
    assert text.estimate_frames(line, slow) == pytest.approx(4 * text.estimate_frames(line, fast), rel=0.05)
    assert text.pace('', 200) == text.DEFAULT_PACE


def test_long_text_is_split_at_sentence_ends():
    sentence = 'This is one sentence of a long paragraph that goes on. '
    pieces = text.split_text((sentence * 30).strip(), text.DEFAULT_PACE, longest=750, piece=375)
    assert len(pieces) > 1 and all(p.endswith('.') for p in pieces)
    assert ' '.join(pieces) == (sentence * 30).strip()
