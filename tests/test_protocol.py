"""The add-on spoken over its wire format, in a real process.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

These run `python -m reverb_addon` and talk JSON lines to it, the way the host
does. They cover every answer the add-on owes the host without the model or
the network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reverb_addon import audio, voices  # noqa: E402


@pytest.fixture()
def proc(tmp_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    env['SUBTITLD_MODELS_DIR'] = str(tmp_path / 'models')
    # Nothing listens on the discard port: no test reaches the network.
    env['HF_ENDPOINT'] = env['REVERB_MODELS_URL'] = 'http://127.0.0.1:9'
    for key in ('REVERB_VOICE_REF_AUDIO', 'REVERB_VOICE_REF_TEXT', 'REVERB_PRECISION'):
        env.pop(key, None)
    process = subprocess.Popen([sys.executable, '-m', 'reverb_addon'], cwd=tmp_path, env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()


def _send(proc, frame: dict) -> None:
    proc.stdin.write((json.dumps(frame) + '\n').encode('utf-8'))
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, proc.stderr.read().decode('utf-8', 'replace')
    return json.loads(line)


def _terminal(proc, req_id: str) -> dict:
    while True:
        frame = _recv(proc)
        if frame.get('id') == req_id and frame['type'] in ('result', 'error'):
            return frame


def _synth(proc, req_id: str, tmp_path: Path, **params) -> dict:
    params.setdefault('output_path', str(tmp_path / f'{req_id}.wav'))
    _send(proc, {'id': req_id, 'type': 'tts.synthesize', 'params': params})
    return _terminal(proc, req_id)


def _clip(tmp_path: Path, seconds: float) -> str:
    path = tmp_path / f'clip-{seconds}.wav'
    audio.write_wav(path, [0.1, -0.1] * int(8000 * seconds), 16000)
    return str(path)


def test_hello_lists_languages_and_voices(proc):
    hello = _recv(proc)
    assert hello['type'] == 'hello' and hello['addon'] == 'reverb' and hello['protocol'] == 1
    [capability] = hello['capabilities']
    assert capability['task'] == 'tts.synthesize'
    assert capability['languages'] == list(voices.LANGUAGES)
    assert capability['voices'] == voices.manifest_voices()
    assert capability['voice_clone'] is True
    assert capability['voices'][0] == {'id': 'reverb-clone', 'language': '*', 'sample_rate': 24000,
                                       'display_name': "Speaker's own voice (cloned from the video)",
                                       'requires': ['voice_ref_audio']}


def test_requests_it_cannot_serve_get_clear_errors(proc, tmp_path):
    _recv(proc)
    _send(proc, {'type': 'ready', 'host': 'subtitld', 'host_version': 'test'})
    preset = voices.presets()[0].id
    assert _synth(proc, 'a', tmp_path, text='Привет.', voice=preset, language='ru')['code'] == 'unsupported_language'
    empty = _synth(proc, 'b', tmp_path, text='Hello.', voice=preset, language='')
    assert empty['code'] == 'unsupported_language' and "project's language" in empty['message']
    assert _synth(proc, 'c', tmp_path, text='Hello.', voice='alba', language='en')['code'] == 'unsupported_voice'
    assert _synth(proc, 'd', tmp_path, text='  ', voice=preset, language='en')['code'] == 'bad_params'
    _send(proc, {'id': 'e', 'type': 'tts.synthesize', 'params': {'text': 'Hi.', 'language': 'en'}})
    assert _terminal(proc, 'e')['code'] == 'bad_params'  # no output_path
    _send(proc, {'id': 'f', 'type': 'asr.transcribe', 'params': {}})
    assert _terminal(proc, 'f')['code'] == 'bad_params'


def test_the_clone_needs_a_clip_and_its_transcript(proc, tmp_path):
    _recv(proc)
    none = _synth(proc, 'a', tmp_path, text='Hello.', voice='reverb-clone', language='en')
    assert none['code'] == 'bad_params' and 'Configure' in none['message']
    gone = _synth(proc, 'b', tmp_path, text='Hello.', voice='reverb-clone', language='en',
                  voice_ref_audio=str(tmp_path / 'missing.wav'), voice_ref_text='Hi there.')
    assert gone['code'] == 'bad_params' and 'missing' in gone['message']
    mute = _synth(proc, 'c', tmp_path, text='Hello.', voice='reverb-clone', language='en',
                  voice_ref_audio=_clip(tmp_path, 4))
    assert mute['code'] == 'bad_params' and 'transcript' in mute['message']


def test_a_download_that_cannot_reach_the_server_is_retryable(proc, tmp_path):
    _recv(proc)
    frame = _synth(proc, 'g', tmp_path, text='Hola.', voice=voices.presets()[0].id, language='es-MX')
    assert frame['code'] == 'network_unavailable' and frame['retryable'] is True


def test_shutdown_exits_cleanly(proc):
    _recv(proc)
    _send(proc, {'type': 'shutdown'})
    assert proc.wait(timeout=10) == 0
