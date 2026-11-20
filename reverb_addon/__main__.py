"""Reverb-TTS add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline text-to-speech with Reverb-TTS v1.0.0 (Software Mansion AI Labs), in
English, German, French, Spanish, Italian, Polish and Portuguese, at 24 kHz,
in 16 preset voices or in the speaker's own voice cloned from a clip and its
transcript. Serves `tts.synthesize`. ONNX Runtime runs graphs converted from
Reverb's weights on the CPU, and numpy runs Reverb's sampler (engine.py). The
files download the first time a line needs them (download.py): the model
and the preset voices, then the codec encoder for a clone. After that nothing
leaves the machine.

The model's weights are CC BY-NC 4.0: for non-commercial use only.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:reverb]` and surfaces.

Options (the host sets them from Configure as REVERB_<KEY>)
-----------------------------------------------------------
threads: CPU threads, 0 for automatic; steps: the sampler's main passes (8,
Reverb's own setting, or fewer for speed); precision: `exact` (the
backbone in full precision) or `fast` (dynamic int8, about twice as fast,
further from the original); voice_ref_audio and
voice_ref_text: a clip and its transcript, which the speaker's own voice
falls back on when the host sends none.

Limits
------
One line at a time. A cancel stops the line between the sampler's passes, or
stops the download (which resumes on the next request).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from reverb_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from reverb_addon import audio, download, tls, voices

ADDON_VERSION = __version__

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_UNSUPPORTED_VOICE = 'unsupported_voice'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'
ERR_OOM = 'out_of_memory'

# No space left on the device, or over the user's disk quota.
_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, 'EDQUOT', 122)}
# Beside a package module: frozen, this script itself runs from _internal/.
_PINS_FILE = Path(download.__file__).resolve().parent / 'pins.json'
# A reference shorter than this gives the model too little of the voice.
_SHORTEST_REFERENCE = 2.0
# How fast a clip's transcript may say its words, against Reverb's usual pace.
# The 16 presets speak at 0.73-1.24; 12 s of a 72 s subtitle sent with all of
# its text came out at 7.1, and Reverb spoke a 25 s line in 3 s of babble; a
# subtitle whose text ran past its audio, at 2.2, gave gibberish too.
_PACE_RANGE = (0.4, 1.6)

log = logging.getLogger('reverb_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`, so nothing else can write a
    stray line into the protocol. Falls back to the Python-level swap if the
    fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _env_number(name: str, default: float, low: float, high: float) -> float:
    raw = _env(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        log.warning('%s=%r is not a number; using %s', name, raw, default)
        value = default
    return max(low, min(high, value))


def _options() -> dict:
    return {
        'threads': int(_env_number('REVERB_THREADS', 0, 0, 256)),
        'steps': int(_env_number('REVERB_STEPS', 8, 2, 32)),
        'precision': 'fast' if _env('REVERB_PRECISION').lower() == 'fast' else 'exact',
        'voice_ref_audio': _env('REVERB_VOICE_REF_AUDIO'),
        'voice_ref_text': _env('REVERB_VOICE_REF_TEXT'),
    }


def _resolve_language(raw) -> str:
    primary = voices.split_tag(raw)
    if primary in voices.LANGUAGES:
        return primary
    spoken = ', '.join(voices.LANGUAGE_NAMES[code] for code in voices.LANGUAGES)
    if not primary:
        raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                          f"Set the project's language first: Reverb speaks {spoken}.")
    raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                      f'Reverb does not speak {raw!r}. It speaks {spoken}; '
                      f'pick another voice engine for other languages.')


def _speed(params: dict) -> float:
    """The host's rate, -100..100 %, as a factor on the take's length: 1.0 is the voice's own pace."""
    try:
        rate = float(params.get('rate') or 0)
    except (TypeError, ValueError):
        rate = 0.0
    return max(0.5, min(2.0, 1.0 + rate / 100.0))


def _seed(params: dict) -> int:
    try:
        return int(params.get('seed', 42)) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return 42


# ---------------------------------------------------------------------------
# ffmpeg, for references that are not PCM WAV
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _read_reference(path: str) -> tuple:
    """(samples, rate): `path` itself if it is PCM WAV, else via ffmpeg."""
    try:
        return audio.read_wav(path)
    except audio.UnreadableAudio:
        pass
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise _AddonError(ERR_BAD_PARAMS,
                          f'The reference {Path(path).name} is not PCM WAV, and no ffmpeg was '
                          f'found to convert it. Use a WAV file.')
    scratch = Path(tempfile.mkdtemp(prefix='reverb-', dir=_env('SUBTITLD_TEMP_DIR') or None))
    try:
        out = scratch / 'reference.wav'
        # The clip's own rate: the add-on resamples as Reverb's package does.
        cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', path,
               '-vn', '-ac', '1', '-c:a', 'pcm_s16le', str(out)]
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=_child_env(),
                              startupinfo=_no_window_startupinfo())
        if done.returncode != 0 or not out.is_file():
            tail = ' / '.join(done.stderr.decode('utf-8', 'replace').strip().splitlines()[-3:])
            raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {tail}')
        try:
            return audio.read_wav(out)
        except audio.UnreadableAudio as exc:
            raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {exc}') from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# The model: downloaded and loaded on first use, shared by every request
# ---------------------------------------------------------------------------
_pins: dict | None = None
_loaded = None              # engine.Reverb
_loaded_key = None
_voices: dict = {}          # preset file name, or (path, size, mtime, transcript) of a clip -> engine.Voice
_model_lock = threading.Lock()


def pins(group: str) -> list[dict]:
    """Every pinned file of a group, each with where it is downloaded from."""
    global _pins
    if _pins is None:
        _pins = json.loads(_PINS_FILE.read_text(encoding='utf-8'))['groups']
    entry = _pins[group]
    source = {key: entry[key] for key in ('repo', 'url', 'revision') if key in entry}
    return [{**pin, **source} for pin in entry['files']]


def group_root(group: str) -> Path:
    """The revision folder a group's files land in."""
    first = pins(group)[0]
    return download.local_path(first).parents[len(first['path'].split('/')) - 1]


def _megabytes(count: int) -> str:
    return f'{count / 1e6:,.0f} MB'


_GROUP_NAMES = {'core': 'the Reverb model', 'backbone': 'its network', 'backbone-fast': 'its fast network',
                'voices': 'the preset voices', 'clone': 'the voice-cloning encoder'}
_BACKBONES = {'exact': ('backbone', 'backbone.onnx'), 'fast': ('backbone-fast', 'backbone_int8.onnx')}


def groups_for(clone: bool, precision: str) -> list[str]:
    """The download groups a line needs."""
    groups = ['core', _BACKBONES[precision][0], 'voices']
    return groups + ['clone'] if clone else groups


def _ensure(state: '_Request', groups: list[str]) -> None:
    """Download whatever of `groups` is missing; progress runs 0 to 0.3."""
    todo = [pin for group in groups for pin in download.missing(pins(group))]
    if not todo:
        return
    total = sum(pin['size'] for pin in todo)
    names = [_GROUP_NAMES[g] for g in groups if download.missing(pins(g))]
    message = f'Downloading {" and ".join(names)} ({_megabytes(total)})...'
    _emit_progress(state.req_id, 0.0, message)
    last = [0.0]

    def progress(done: int, total_bytes: int) -> None:
        now = time.monotonic()
        if now - last[0] >= 0.25 or done >= total_bytes:
            last[0] = now
            _emit_progress(state.req_id, 0.3 * done / max(1, total_bytes), message)

    started = time.monotonic()
    download.ensure(todo, progress, state.cancel.is_set)
    log.info('downloaded %s in %.1fs', _megabytes(total), time.monotonic() - started)


def _load(state: '_Request', options: dict, groups: list[str]):
    """Download (once) and load. The caller holds _model_lock."""
    global _loaded, _loaded_key
    from reverb_addon import engine  # numpy and onnxruntime: only once a line comes

    _ensure(state, groups)
    threads = options['threads'] or engine.default_threads()
    group, name = _BACKBONES[options['precision']]
    backbone = group_root(group) / name
    encoder = group_root('clone') / 'codec_encoder.onnx'
    key = (threads, options['precision'])
    if _loaded is None or _loaded_key != key:
        _emit_progress(state.req_id, 0.3, 'Loading Reverb...')
        started = time.monotonic()
        _loaded = None
        _voices.clear()
        _loaded = engine.Reverb(group_root('core'), backbone, threads=threads, encoder_path=encoder)
        _loaded_key = key
        log.info('loaded the %s model with %d threads in %.1fs', options['precision'], threads,
                 time.monotonic() - started)
    return _loaded


def _voice(model, voice: voices.Voice, reference: str, transcript: str, language: str):
    """The engine voice of a preset, or of a clip and its transcript, cached."""
    from reverb_addon import engine, text

    if voice.id != voices.CLONE:
        if voice.file not in _voices:
            _voices[voice.file] = engine.load_preset(group_root('voices') / 'voices' / voice.file)
        return _voices[voice.file]
    stat = os.stat(reference)
    key = (os.path.abspath(reference), stat.st_size, stat.st_mtime_ns, transcript)
    if key in _voices:
        return _voices[key]
    samples, rate = _read_reference(reference)
    seconds = len(samples) / rate
    if seconds < _SHORTEST_REFERENCE:
        raise _AddonError(ERR_BAD_PARAMS, f'The reference clip is {seconds:.1f} s long; Reverb needs at least '
                                          f'{_SHORTEST_REFERENCE:.0f} s, and 3 to 10 s of one person speaking '
                                          f'works best.')
    if seconds > engine.LONG_REFERENCE:
        log.warning('the reference is %.1f s long; references over %.0f s tend to make worse takes',
                    seconds, engine.LONG_REFERENCE)
    # The transcript is verbalized in its own language, which the host does not send.
    spoken = text.guess_language(transcript, language)
    made = model.clone_voice(samples, rate, transcript, spoken)
    pace = text.pace(made.text, made.pace_frames) / text.DEFAULT_PACE
    if not _PACE_RANGE[0] <= pace <= _PACE_RANGE[1]:
        # Reverb paces the take by the transcript and goes on from its last words.
        more_or_less = 'more' if pace > 1 else 'fewer'
        raise _AddonError(ERR_BAD_PARAMS,
                          f"The clip's transcript has far {more_or_less} words than the clip says "
                          f'({pace:.1f} times the usual pace), so Reverb would make a broken take. Send the '
                          'words spoken in the clip, and only those, or pick a preset voice.')
    log.info('cloned a %.1f s reference (%d frames, %d of speech), transcript in %s',
             seconds, made.codes.shape[1], made.pace_frames, spoken)
    clips = [k for k in _voices if isinstance(k, tuple)]
    while len(clips) >= 16:
        _voices.pop(clips.pop(0))
    _voices[key] = made
    return made


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _clone_reference(params: dict, options: dict) -> tuple[str, str]:
    """(clip, transcript) for the speaker's own voice: the request's, else Configure's."""
    reference = str(params.get('voice_ref_audio') or '')
    transcript = ' '.join(str(params.get('voice_ref_text') or '').split())
    if not reference:
        reference, transcript = options['voice_ref_audio'], ' '.join(options['voice_ref_text'].split())
    if not reference:
        raise _AddonError(ERR_BAD_PARAMS,
                          "The speaker's own voice needs a clip of it, and none came with the request. "
                          'Pick a reference clip and its transcript in Configure, or a preset voice.')
    if not os.path.isfile(reference):
        raise _AddonError(ERR_BAD_PARAMS, f'The reference clip is missing: {reference}')
    if not transcript:
        # Without it Reverb drops the first words of every line.
        raise _AddonError(ERR_BAD_PARAMS,
                          "Reverb clones from a clip and what is said in it, and the clip's transcript is "
                          "missing. Subtitld sends it when the speaker's subtitles are in the language of the "
                          "audio (set the project's source language if you translated them). When it has to "
                          "cut the clip from a longer subtitle, it also needs a speech recognition add-on for "
                          "that language, to find the words in the cut. For the clip in Configure, type its "
                          'transcript there. Or pick a preset voice.')
    return reference, transcript


def _handle_synthesize(state: _Request, params: dict) -> None:
    req_id = state.req_id
    line = ' '.join(str(params.get('text') or '').split())
    output_path = params.get('output_path')
    if not line:
        _emit_error(req_id, ERR_BAD_PARAMS, 'text is empty')
        return
    if not isinstance(output_path, str) or not output_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'output_path is required')
        return
    options = _options()
    try:
        language = _resolve_language(params.get('language'))
        voice_id = str(params.get('voice') or '') or voices.CLONE
        voice = voices.by_id(voice_id)
        if voice is None:
            raise _AddonError(ERR_UNSUPPORTED_VOICE, f'unknown voice {voice_id!r}')
        reference, transcript = _clone_reference(params, options) if voice.id == voices.CLONE else ('', '')
    except _AddonError as exc:
        _emit_error(req_id, exc.code, exc.message)
        return
    seed = _seed(params)
    speed = _speed(params)

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        from reverb_addon import engine
        if state.cancel.is_set():
            raise engine.Cancelled()
        with _model_lock:
            model = _load(state, options, groups_for(voice.id == voices.CLONE, options['precision']))
            if state.cancel.is_set():
                raise engine.Cancelled()
            spoken = _voice(model, voice, reference, transcript, language)
            _emit_progress(req_id, 0.35, 'Speaking...')

            def progress(fraction: float) -> None:
                _emit_progress(req_id, 0.35 + 0.6 * fraction, 'Speaking...')

            log.info('synthesize %s: voice=%s language=%s seed=%d speed=%.2f steps=%d precision=%s chars=%d',
                     req_id, voice.id, language, seed, speed, options['steps'], options['precision'], len(line))
            started = time.monotonic()
            wav = model.synthesize(line, spoken, language, speed=speed, seed=seed,
                                   sampler=engine.Sampler(steps=options['steps']),
                                   progress=progress, cancelled=state.cancel.is_set)
            took = time.monotonic() - started
        if state.cancel.is_set():
            raise engine.Cancelled()
        if not len(wav):
            raise _AddonError(ERR_BAD_PARAMS, 'Nothing in the line could be spoken.')
        audio.write_wav(Path(output_path), wav, model.sample_rate)
        duration = len(wav) / model.sample_rate
        log.info('synthesize %s: %.2fs of audio in %.2fs', req_id, duration, took)
        outcome = ('result', {
            'path': output_path,
            'duration_sec': round(duration, 3),
            'sample_rate': model.sample_rate,
            'channels': 1,
            'voice': voice.id,
        })
    except (download.Cancelled, _cancelled_type()):
        log.info('synthesize %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except download.DownloadError as exc:
        log.warning('synthesize %s: download failed: %s', req_id, exc)
        outcome = ('error', ERR_NETWORK, f'Could not download the model: {exc}', True)
    except _AddonError as exc:
        log.warning('synthesize %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, 'Out of memory. Reverb needs about 1.5 GB free.', False)
    except OSError as exc:
        log.exception('synthesize %s failed', req_id)
        if getattr(exc, 'errno', None) in _DISK_FULL_ERRNOS:
            outcome = ('error', ERR_DISK_FULL, f'Out of disk space: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('synthesize %s failed', req_id)
        outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _cancelled_type():
    from reverb_addon import engine
    return engine.Cancelled


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'reverb-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'tts.synthesize',
            'languages': list(voices.LANGUAGES),
            'voices': voices.manifest_voices(),
            'voice_clone': True,
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'tts.synthesize':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('tts.synthesize without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_synthesize, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Check what the frozen bundle needs, without the network or the model.

    ONNX Runtime must load and run a graph; the tokenizer library and Verbalis
    must work; the resampler must run; and the TLS and hashing modules the
    downloads use must be there.
    """
    import ssl

    import numpy as np
    import onnxruntime as ort
    from onnxruntime import InferenceSession
    from tokenizers import Tokenizer, models, pre_tokenizers

    from reverb_addon import engine, text  # noqa: F401  (engine: its imports must resolve)

    session = InferenceSession(_tiny_graph(), providers=['CPUExecutionProvider'])
    doubled = session.run(None, {'x': np.ones(3, dtype=np.float32)})[0]
    tokenizer = Tokenizer(models.WordLevel({'hello': 0, 'world': 1, '[UNK]': 2}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    ids = tokenizer.encode('hello world').ids
    spoken = text.prepare('I have 3 cats.', 'en')
    resampled = audio.resample(np.zeros(16000, dtype=np.float32), 16000, 24000)
    ok = (bool(np.allclose(doubled, 2)) and ids == [0, 1] and 'three' in spoken and len(resampled) == 24000
          and text.guess_language('Ich habe die Katze nicht gesehen.', 'en') == 'de')
    report = {
        'ok': ok,
        'version': ADDON_VERSION,
        'onnxruntime': ort.__version__,
        'numpy': np.__version__,
        'voices': len(voices.all_voices()),
        'pins': {group: len(pins(group)) for group in ('core', 'backbone', 'backbone-fast', 'clone', 'voices')},
        'spoken': spoken,
        'openssl': ssl.OPENSSL_VERSION,
        'sha256': hashlib.sha256(b'').hexdigest()[:8],
        'ca_bundle': os.environ.get('SSL_CERT_FILE', ''),
    }
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0 if ok else 1


def _tiny_graph() -> bytes:
    """A serialized ONNX model computing x + x, written without the onnx package."""
    def varint(n: int) -> bytes:
        out = b''
        while True:
            byte, n = n & 0x7F, n >> 7
            out += bytes([byte | (0x80 if n else 0)])
            if not n:
                return out

    def field(number: int, payload: bytes) -> bytes:  # length-delimited
        return varint(number << 3 | 2) + varint(len(payload)) + payload

    def number(field_no: int, value: int) -> bytes:   # varint
        return varint(field_no << 3) + varint(value)

    node = field(1, b'x') + field(1, b'x') + field(2, b'y') + field(4, b'Add')
    dim = field(1, number(1, 3))                                  # TensorShapeProto.Dimension
    tensor = number(1, 1) + field(2, dim)                         # elem_type FLOAT, shape
    value_type = field(1, tensor)                                 # TypeProto.tensor_type

    def value_info(name: bytes) -> bytes:
        return field(1, name) + field(2, value_type)

    graph = field(1, node) + field(2, b'g') + field(11, value_info(b'x')) + field(12, value_info(b'y'))
    opset = field(1, b'') + number(2, 13)
    return number(1, 8) + field(7, graph) + field(8, opset)


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'reverb-addon {ADDON_VERSION}')
        return 0
    # Lines and transcripts quote accented text; a Windows stderr is cp1252.
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(
        level=(_env('REVERB_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('reverb-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        _stop_workers()
    return rc


if __name__ == '__main__':
    sys.exit(main())
