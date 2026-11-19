"""Audio in and out: PCM WAV with the standard library, resampling, levels.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Subtitld's own reference clips are 16 kHz mono PCM WAV. Anything else is
converted with ffmpeg first (see __main__).

The resampler is torchaudio's windowed-sinc one, in numpy, so a reference is
encoded from the samples Reverb's own package would hand the codec: Kaiser
"best" (librosa's kaiser_best) to 24 kHz, as the training references were,
then the Hann default to 16 kHz inside the codec. `raise_quiet`,
`trim_silence` and `speech_length` are reverb_tts.audio's (Apache-2.0,
Copyright Software Mansion), unchanged but for torch.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

# reverb_tts.audio.RESAMPLE: librosa's kaiser_best.
KAISER_BEST = {'lowpass_filter_width': 64, 'rolloff': 0.9475937167399596, 'window': 'kaiser',
               'beta': 14.769656459379392}
# torchaudio.functional.resample's defaults, which the codec uses for its 16 kHz copy.
HANN_DEFAULT = {'lowpass_filter_width': 6, 'rolloff': 0.99, 'window': 'hann', 'beta': None}

SILENCE_DB = -50.0  # a 10 ms window quieter than this (dBFS) is silence
QUIET_DB = -40.0    # a reference quieter than this (RMS) is raised to LEVEL_DB before it is encoded
LEVEL_DB = -20.0
PEAK = 0.99         # the raise stops short of taking the loudest sample past this


class UnreadableAudio(Exception):
    """Not a PCM WAV file this reader takes."""


def read_wav(path) -> tuple[np.ndarray, int]:
    """(mono samples as float32 in [-1, 1], sample rate)."""
    try:
        with wave.open(str(path), 'rb') as clip:
            width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
            data = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise UnreadableAudio(f'{path}: {exc}') from exc
    if width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (np.where(value >= 1 << 23, value - (1 << 24), value) / float(1 << 23)).astype(np.float32)
    elif width == 4:
        samples = (np.frombuffer(data, dtype='<i4').astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise UnreadableAudio(f'{path}: {8 * width}-bit samples')
    if not len(samples):
        raise UnreadableAudio(f'{path}: no audio')
    return samples.reshape(-1, channels).mean(axis=1), rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """16-bit mono PCM, written beside `path` and renamed into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype('<i2')
    with wave.open(str(part), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    part.replace(path)


def _sinc_kernel(orig: int, new: int, lowpass_filter_width: int, rolloff: float, window: str,
                 beta: float | None) -> tuple[np.ndarray, int]:
    """torchaudio's `_get_sinc_resample_kernel`: [new, taps] filters and the padding width."""
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx) * base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    if window == 'hann':
        weights = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        weights = np.i0(beta * np.sqrt(1 - (t / lowpass_filter_width) ** 2)) / np.i0(beta)
    t = t * math.pi
    with np.errstate(divide='ignore', invalid='ignore'):
        kernels = np.where(t == 0, 1.0, np.sin(t) / t)
    return (kernels * weights * (base_freq / orig)).astype(np.float32), width


def resample(samples: np.ndarray, orig_rate: int, new_rate: int, settings: dict = KAISER_BEST) -> np.ndarray:
    """torchaudio.functional.resample on mono float32 samples."""
    if orig_rate == new_rate:
        return np.asarray(samples, dtype=np.float32)
    gcd = math.gcd(int(orig_rate), int(new_rate))
    orig, new = int(orig_rate) // gcd, int(new_rate) // gcd
    kernel, width = _sinc_kernel(orig, new, **settings)
    x = np.asarray(samples, dtype=np.float32)
    length = len(x)
    padded = np.pad(x, (width, width + orig))
    taps = kernel.shape[1]
    count = (len(padded) - taps) // orig + 1
    frames = np.lib.stride_tricks.as_strided(padded, shape=(count, taps),
                                             strides=(padded.strides[0] * orig, padded.strides[0]))
    out = (frames @ kernel.T).reshape(-1)
    return out[: math.ceil(new * length / orig)].astype(np.float32)


def raise_quiet(samples: np.ndarray) -> np.ndarray:
    """Raised to LEVEL_DB if quieter than QUIET_DB (RMS), the peak kept at PEAK; otherwise as it is."""
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0
    if rms == 0.0 or 20 * np.log10(rms) >= QUIET_DB:
        return samples
    return (samples * min(10 ** (LEVEL_DB / 20) / rms, PEAK / float(np.abs(samples).max()))).astype(np.float32)


def _levels(samples: np.ndarray, window: int) -> np.ndarray:
    """The level of every whole `window` of samples, in dBFS."""
    count = len(samples) // window
    return 20 * np.log10(np.sqrt((samples[: count * window].reshape(count, window) ** 2).mean(1)) + 1e-10)


def trim_silence(samples: np.ndarray, rate: int, keep: float = 0.1) -> np.ndarray:
    """Cuts leading and trailing silence (10 ms windows under SILENCE_DB), keeping `keep` seconds of it each side."""
    window = rate // 100
    level = _levels(samples, window)
    loud = np.flatnonzero(level >= SILENCE_DB)
    if len(loud) == 0:
        return samples
    count = len(level)
    sounds = np.split(loud, np.flatnonzero(np.diff(loud) > 20) + 1)  # separated by over 200 ms of silence
    # A sound of at most 120 ms, alone at either edge, is a codec burst rather than speech.
    if len(sounds) > 1 and len(sounds[0]) <= 12:
        sounds = sounds[1:]
    if len(sounds) > 1 and len(sounds[-1]) <= 12:
        sounds = sounds[:-1]
    margin = round(keep * 100)
    return samples[max(sounds[0][0] - margin, 0) * window: min(sounds[-1][-1] + 1 + margin, count) * window]


def speech_length(samples: np.ndarray, rate: int, max_pause: float = 0.5, lead: float = 0.1,
                  trail: float = 0.2) -> float:
    """Seconds of a reference that set the speaker's pace: without silent edges, each pause at most `max_pause`."""
    window = rate // 100
    count = len(samples) // window
    if count == 0:
        return len(samples) / rate
    x = np.asarray(samples[: count * window], dtype=np.float64)
    level = _levels(x * 0.1 / max(float(np.sqrt(np.mean(x ** 2))), 1e-8), window)
    loud = np.flatnonzero(level >= SILENCE_DB)
    if len(loud) == 0:
        return len(samples) / rate
    first, last = int(loud[0]), int(loud[-1])
    gaps = np.diff(loud) - 1  # silent windows between consecutive loud ones
    kept = len(loud) + int(np.minimum(gaps, round(max_pause * 100)).sum())
    edges = min(first, round(lead * 100)) + min(count - 1 - last, round(trail * 100))
    return (kept + edges) / 100
