"""Reverb on ONNX Runtime and numpy: no torch, no transformers.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The graphs are made from Reverb v1.0.0's weights by tools/export_models.py:

- backbone.onnx: the Qwen3 backbone, 24 layers, its weights stored as fp16
  and computed in fp32; backbone_int8.onnx, the same in dynamic int8, about
  twice as fast and further from the original. One graph does both jobs. With an empty cache it reads the
  prefix (prompt text, then the reference's frames) and returns that prefix's
  keys and values. With the prefix's cache it reads the target frames and
  returns their hidden states. The prefix never attends to the target, so it
  runs once a take.
- codec_decoder.onnx and codec_encoder.onnx: the Higgs Audio v2 codec, codes
  to 24 kHz audio and back. The encoder is only needed to clone.
- tables.npz: the text and audio embeddings and the audio heads, which numpy
  applies around the backbone.

The sampler is reverb_tts.sampler's (Apache-2.0, Copyright Software Mansion),
in numpy, for one take at a time. Its 8 main passes commit codebooks 0-1, a few
slots a pass with Gumbel noise, and withdraw the commits the model has come to
doubt. Then 2 polish passes fill codebooks 2-7, leaning towards the
reference's frames. Its noise comes from numpy rather than torch, so a seed
gives a different take than the package does, an equally good one.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from reverb_addon import audio, text

SAMPLE_RATE = 24000
FRAME_RATE = 25       # codec frames a second
HOP = 960             # samples a codec frame
CODEBOOKS = 8
CODES = 1025          # entries a codebook: 1024 codes and the mask
MASK = 1024
LAYERS, KV_HEADS, HEAD_DIM = 24, 6, 128
END_CUT = 0.2         # seconds cut from every take's end, where the codec can leave a short burst of noise
LONGEST, PIECE = 30.0, 15.0  # a text estimated over LONGEST seconds is spoken in pieces of about PIECE
LONG_REFERENCE = 15.0  # seconds: references longer than this are known to hurt


class Cancelled(Exception):
    """The request was cancelled."""


@dataclass(frozen=True)
class Sampler:
    """reverb_tts.sampler.SamplerConfig's tuned defaults."""
    steps: int = 8
    remask_ratio: float = 0.25
    t_shift: float = 0.1
    codebook_penalty: float = 5.0
    position_temperature: float = 5.0
    polish_bias: float = 1.0
    polish_guidance: float = 0.5


@dataclass
class Voice:
    """A reference: its codes [8, frames], its prepared transcript, and how many of its frames are speech."""
    codes: np.ndarray
    text: str
    speech_frames: int | None = None

    @property
    def pace_frames(self) -> int:
        frames = self.codes.shape[1]
        return frames if self.speech_frames is None else max(1, min(self.speech_frames, frames))


def default_threads() -> int:
    """At most 8: more only oversubscribes."""
    return max(1, min(8, os.cpu_count() or 1))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _rank(score: np.ndarray, descending: bool) -> np.ndarray:
    """Each entry's rank within the flat array."""
    order = np.argsort(-score if descending else score, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def _prompt(reference_text: str, part: str, language: str) -> str:
    return (f'<|denoise|><|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|>'
            f'<|text_start|>{reference_text} {part}<|text_end|>')


def load_preset(path: Path) -> Voice:
    """A preset voice file of the model repository: {codes, transcript, language, speech_frames}."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    codes = np.asarray(data['codes'], dtype=np.int64).reshape(CODEBOOKS, -1)
    if codes.size == 0 or codes.min() < 0 or codes.max() >= MASK:
        raise ValueError(f'{path}: codes out of range')
    return Voice(codes, data['transcript'], data.get('speech_frames'))


class Reverb:
    """The loaded model. `root` holds the core files; the encoder loads on the first clone."""

    def __init__(self, root: Path, backbone_path: Path, threads: int, encoder_path: Path | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.root = Path(root)
        self.threads = threads
        self._ort = ort
        self.backbone = self._session(Path(backbone_path))
        self.decoder = self._session(self.root / 'codec_decoder.onnx')
        self.encoder_path = encoder_path
        self.encoder = None
        with np.load(self.root / 'tables.npz') as tables:
            self.text_table = tables['text_table'].astype(np.float32)
            self.text_project = tables['text_project'].astype(np.float32)
            self.audio_embeddings = tables['audio_embeddings'].astype(np.float32)
            self.audio_heads = tables['audio_heads'].astype(np.float32)
        self.tokenizer = Tokenizer.from_file(str(self.root / 'tokenizer.json'))
        self.sample_rate = SAMPLE_RATE
        self._offsets = (CODES * np.arange(CODEBOOKS))[:, None]

    def _session(self, path: Path):
        options = self._ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        return self._ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    # -- voices ---------------------------------------------------------------
    def clone_voice(self, samples: np.ndarray, rate: int, transcript: str, language: str) -> Voice:
        """A voice from a recording (mono float32 at any rate) and its transcript, in `language`."""
        if self.encoder is None:
            if self.encoder_path is None:
                raise RuntimeError('no codec encoder to clone with')
            self.encoder = self._session(self.encoder_path)
        wav = audio.raise_quiet(audio.resample(samples, rate, SAMPLE_RATE, audio.KAISER_BEST))
        # Whole codec frames, as the training references were: otherwise the codes can come out misaligned.
        wav = np.pad(wav, (0, -len(wav) % HOP)).astype(np.float32)
        wav16 = audio.resample(wav, SAMPLE_RATE, 16000, audio.HANN_DEFAULT)
        codes = self.encoder.run(None, {'wav24': wav[None, None], 'wav16': wav16[None]})[0][0].astype(np.int64)
        speech = round(audio.speech_length(wav, SAMPLE_RATE) * FRAME_RATE)
        prepared = text.prepare(transcript, language) if transcript.strip() else ''
        return Voice(codes, prepared, speech)

    # -- speech ---------------------------------------------------------------
    def plan(self, line: str, voice: Voice, language: str, speed: float = 1.0) -> list[tuple[str, int]]:
        """The pieces a line is spoken in, each with its length in frames."""
        words = text.prepare(line, language)
        speaker_pace = text.pace(voice.text, voice.pace_frames)
        parts = text.split_text(words, speaker_pace, int(LONGEST * FRAME_RATE), int(PIECE * FRAME_RATE))
        return [(part, max(1, round(text.estimate_frames(part, speaker_pace) / speed))) for part in parts if part]

    def synthesize(self, line: str, voice: Voice, language: str, speed: float = 1.0, seed: int = 42,
                   sampler: Sampler = Sampler(), progress: Callable[[float], None] | None = None,
                   cancelled: Callable[[], bool] = lambda: False) -> np.ndarray:
        """Mono float32 samples at 24 kHz."""
        pieces = self.plan(line, voice, language, speed)
        passes = sampler.steps + 3  # the prefix, the main passes and the two polish passes
        total = max(1, passes * len(pieces))
        done = [0]

        def step() -> None:
            if cancelled():
                raise Cancelled()
            done[0] += 1
            if progress is not None:
                progress(min(0.99, done[0] / total))

        takes = []
        for index, (part, frames) in enumerate(pieces):
            ids = np.asarray(self.tokenizer.encode(_prompt(voice.text, part, language)).ids, dtype=np.int64)
            codes = self._sample(ids, voice.codes, frames, (seed + index) & 0xFFFFFFFF, sampler, step)
            wav = self.decoder.run(None, {'codes': codes[None]})[0][0, 0]
            wav = wav[: max(len(wav) - int(END_CUT * SAMPLE_RATE), 0)]
            takes.append(audio.trim_silence(wav, SAMPLE_RATE))
        return np.concatenate(takes).astype(np.float32) if takes else np.zeros(0, dtype=np.float32)

    def _embed_audio(self, codes: np.ndarray) -> np.ndarray:
        """[8, T] codes -> [T, hidden]: the sum of each frame's codebook embeddings."""
        return self.audio_embeddings[codes + self._offsets].sum(0)

    def _sample(self, ids: np.ndarray, reference: np.ndarray, frames: int, seed: int, cfg: Sampler,
                step: Callable[[], None]) -> np.ndarray:
        prefix = np.concatenate([self.text_table[ids] @ self.text_project.T, self._embed_audio(reference)])
        P, R, F = prefix.shape[0], reference.shape[1], frames
        empty = np.zeros((LAYERS, 1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        _, keys, values = self.backbone.run(None, {
            'embeds': prefix[None].astype(np.float32), 'mask': np.zeros((1, 1, P, P), np.float32),
            'positions': np.arange(P, dtype=np.int64)[None], 'past_keys': empty, 'past_values': empty})
        step()
        plain = np.zeros((1, 1, F, P + F), np.float32)
        polish = plain.copy()
        polish[..., P - R:P] += cfg.polish_bias
        positions = (P + np.arange(F, dtype=np.int64))[None]

        def log_probs(tokens: np.ndarray, first: int, last: int, mask: np.ndarray) -> np.ndarray:
            hidden = self.backbone.run(['hidden'], {
                'embeds': self._embed_audio(tokens)[None].astype(np.float32), 'mask': mask,
                'positions': positions, 'past_keys': keys, 'past_values': values})[0][0]
            logits = hidden @ self.audio_heads[first * CODES: last * CODES].T
            out = _log_softmax(logits.reshape(F, last - first, CODES).transpose(1, 0, 2))[..., :MASK]
            step()
            return out

        rng = np.random.default_rng(seed)
        tokens = np.full((CODEBOOKS, F), MASK, dtype=np.int64)
        penalty = cfg.codebook_penalty * np.arange(2)[:, None]
        t = np.linspace(0, 1, cfg.steps + 1)
        t = cfg.t_shift * t / (1 - (1 - cfg.t_shift) * t)
        for index in range(cfg.steps):
            lp = log_probs(tokens, 0, 2, plain)
            confidence, prediction = lp.max(-1), lp.argmax(-1)
            held = tokens[:2]
            masked, revealed = held == MASK, held != MASK
            left, last = int(masked.sum()), index == cfg.steps - 1
            # This pass's share of the 2 * frames slots; the last pass fills whatever is left.
            k = left if last else min(left, math.ceil(2 * F * (t[index + 1] - t[index])))
            gumbel = -np.log(-np.log(rng.random((2, F)) + 1e-10) + 1e-10)
            score = np.where(masked, (confidence - penalty) / cfg.position_temperature + gumbel, -np.inf).reshape(-1)
            chosen = ((_rank(score, True) < k) & np.isfinite(score)).reshape(2, F)
            updated = np.where(chosen, prediction, held)
            if not last:
                # Withdraw the held codes the model now finds least likely, remask_ratio as many as this pass committed.
                likely = np.take_along_axis(lp, np.minimum(held, MASK - 1)[..., None], -1)[..., 0]
                count = min(round(cfg.remask_ratio * k), int(revealed.sum()))
                doubted = (_rank(np.where(revealed, likely, np.inf).reshape(-1), False) < count).reshape(2, F)
                updated = np.where(doubted & revealed, MASK, updated)
            tokens[:2] = updated
        # Polish: codebooks 2-3, then 4-7 guided against the first polish pass's view of them.
        first = log_probs(tokens, 2, 8, polish)
        tokens[2:4] = first[:2].argmax(-1)
        second = log_probs(tokens, 4, 8, polish)
        tokens[4:] = (second + cfg.polish_guidance * (second - first[2:])).argmax(-1)
        return tokens
