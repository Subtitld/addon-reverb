"""The voices the add-on offers, and the languages Reverb speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

- `reverb-clone` clones the speaker from the video: the host hands over a
  clip of their speech with its transcript, and the codec encoder turns the
  clip into the codes Reverb continues from.
- The 16 preset voices are the model repository's voices/<name>.json: codes
  and transcript, ready to use. They download with the model. Every voice
  speaks all seven languages, with the accent of the language it was
  recorded in.
"""

from __future__ import annotations

from dataclasses import dataclass

CLONE = 'reverb-clone'
LANGUAGES = ('en', 'de', 'fr', 'es', 'it', 'pl', 'pt')
LANGUAGE_NAMES = {'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian',
                  'pl': 'Polish', 'pt': 'Portuguese'}

SAMPLE_RATE = 24000

# (preset file stem, display name)
_PRESETS = (
    ('en_female_narrator', 'English woman, narrator'),
    ('en_female_whisper', 'English woman, whispering'),
    ('en_male_narrator_1', 'English man, narrator 1'),
    ('en_male_narrator_2', 'English man, narrator 2'),
    ('en_male_angry', 'English man, angry'),
    ('de_female_standard', 'German woman'),
    ('de_male_standard', 'German man'),
    ('fr_female_standard', 'French woman'),
    ('fr_male_standard', 'French man'),
    ('es_male_standard_1', 'Spanish man 1'),
    ('es_male_standard_2', 'Spanish man 2'),
    ('it_male_standard', 'Italian man'),
    ('it_male_expressive', 'Italian man, expressive'),
    ('pl_male_standard', 'Polish man'),
    ('pt_female_standard', 'Portuguese woman'),
    ('pt_male_standard', 'Portuguese man'),
)


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    file: str = ''   # voices/<file> in the model repository, empty for the clone


def presets() -> tuple[Voice, ...]:
    return tuple(Voice('reverb-' + stem.replace('_', '-'), name, f'{stem}.json') for stem, name in _PRESETS)


def all_voices() -> tuple[Voice, ...]:
    return (Voice(CLONE, "Speaker's own voice (cloned from the video)"), *presets())


def by_id(voice_id: str) -> Voice | None:
    return next((v for v in all_voices() if v.id == voice_id), None)


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in all_voices():
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name, 'sample_rate': SAMPLE_RATE}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out


def split_tag(tag) -> str:
    """`pt-BR` -> 'pt'. Lower-cased; '' when there is none."""
    return str(tag or '').strip().lower().replace('_', '-').split('-')[0]
