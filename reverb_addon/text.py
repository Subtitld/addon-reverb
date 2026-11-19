"""Text as Reverb reads it, and how long a take should be.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

`prepare`, `estimate_frames` and `split_text` are reverb_tts.text's
(Apache-2.0, Copyright Software Mansion): Verbalis spells numbers, dates,
money, units and symbols out as words, the way the training transcripts are
written, and a take's length follows the reference speaker's pace.

`guess_language` is the add-on's own. The host sends the reference's
transcript but not its language, and the transcript has to be verbalized in
the language it is written in: "12" is "twelve" in an English transcript
even when the line is Portuguese.
"""

from __future__ import annotations

import functools
import re
import unicodedata

from verbalis import Verbalizer

LANGUAGES = {'de': 'German', 'en': 'English', 'es': 'Spanish', 'fr': 'French', 'it': 'Italian', 'pl': 'Polish',
             'pt': 'Portuguese'}

BRACKETS = re.compile(r'[()\[\]{}]')  # never in the training transcripts
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
# With no transcript to measure the speaker by: the median pace of the preset voices, in text weight per frame.
DEFAULT_PACE = 0.49

# Frequent short words that tell the seven languages apart.
_STOPWORDS = {
    'en': 'the and is are you that this with have was what for not it of to',
    'de': 'der die das und ist nicht ich du sie es ein eine mit auf den dem zu',
    'es': 'el la los las y es que de no por con para una un se lo',
    'fr': 'le la les et est que de ne pas je tu il des un une pour avec',
    'it': 'il la le e che di non per con una un sono è gli lo io',
    'pl': 'i w nie się na to że jest z do jak ale co tak',
    'pt': 'o a os as e que de não para com uma um é do da em',
}
_STOPWORD_SETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
_LETTERS = {'pl': set('ąćęłńśźż'), 'de': set('äöüß'), 'pt': set('ãõ'), 'es': set('ñ¿¡'), 'fr': set('œëîïûù')}


@functools.lru_cache(maxsize=None)
def _verbalizer(language: str) -> Verbalizer:
    return Verbalizer(language)


def prepare(text: str, language: str) -> str:
    """Verbalized, with brackets dropped and single spaces."""
    text = _verbalizer(language).process(text)
    return ' '.join(BRACKETS.sub('', text).split())


def guess_language(text: str, fallback: str) -> str:
    """The most likely of the seven languages for `text`, or `fallback` when nothing tells."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    scores = {lang: sum(word in stop for word in words) for lang, stop in _STOPWORD_SETS.items()}
    for lang, letters in _LETTERS.items():
        scores[lang] += 2 * sum(char in letters for char in lowered)
    best = max(scores.values())
    if best == 0:
        return fallback
    leaders = [lang for lang, score in scores.items() if score == best]
    return fallback if fallback in leaders else leaders[0]


def _weight(string: str) -> float:
    kinds = (unicodedata.category(char)[0] for char in string)
    return sum({'Z': 0.2, 'M': 0.0, 'P': 0.5, 'S': 0.5, 'N': 3.5}.get(kind, 1.0) for kind in kinds)


def pace(reference_text: str, reference_frames: int) -> float:
    """Text weight per frame at which the reference speaks: DEFAULT_PACE without a transcript."""
    weight = _weight(reference_text)
    return weight / reference_frames if weight > 0 and reference_frames > 0 else DEFAULT_PACE


def estimate_frames(text: str, speaker_pace: float) -> int:
    """The take's length in codec frames: the text's spoken weight at the speaker's pace."""
    frames = _weight(text) / speaker_pace
    if frames < 50:  # short texts get relatively more time
        frames = 50 * (frames / 50) ** (1 / 3)
    return max(1, int(frames))


def split_text(text: str, speaker_pace: float, longest: int, piece: int) -> list[str]:
    """The text whole when it is at most `longest` frames, else cut at sentence ends into about `piece` frames."""
    if estimate_frames(text, speaker_pace) <= longest:
        return [text]
    pieces: list[str] = []
    for sentence in SENTENCE_END.split(text):
        joined = f'{pieces[-1]} {sentence}' if pieces else sentence
        if pieces and estimate_frames(joined, speaker_pace) <= piece:
            pieces[-1] = joined
        else:
            pieces.append(sentence)
    return pieces
