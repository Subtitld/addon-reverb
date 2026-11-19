# Reverb-TTS add-on for Subtitld

Offline **text-to-speech with voice cloning** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by
[Reverb-TTS](https://github.com/software-mansion-labs/Reverb-TTS) v1.0.0 by
Software Mansion AI Labs, run on [ONNX Runtime](https://onnxruntime.ai).

**The model is licensed under CC BY-NC 4.0: for non-commercial use only.**
The add-on itself is Apache-2.0.

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **Seven languages.** English, German, French, Spanish, Italian, Polish and
  Portuguese, at 24 kHz. Numbers, dates, prices and units are read as words
  ([Verbalis](https://github.com/software-mansion-labs/Verbalis)).
- **16 preset voices, or the speaker's own.** Subtitld sends a clip of the
  speaker from the video with the subtitles spoken in it, and the dub speaks
  in their voice. Every voice speaks every language.
- **CPU only.** No GPU, no torch. On an i7-13700T that was busy with other
  work, about 1.2 times faster than real time (Exact) or 1.6 times (Fast),
  with about 1.5 GB of memory.
- **Offline after the first line.** Each file downloads the first time a line
  needs it, then nothing leaves the machine: no telemetry, no account.

## Voices

| Voice | |
|-------|---|
| Speaker's own voice | Clones the clip of the speaker Subtitld sends with each line, or the fallback clip in Configure |
| English woman (narrator, whispering), English man (narrator 1 and 2, angry) | English presets |
| German woman, German man; French woman, French man | |
| Spanish man 1 and 2; Italian man (and expressive); Polish man | |
| Portuguese woman, Portuguese man | |

The presets are Reverb's own (`voices/*.json` in its repository): codes and a
transcript, nothing to encode. A preset speaks other languages with the accent
of the one it was recorded in.

**Cloning needs the clip's transcript.** Reverb continues from the reference
and its words; without them it drops the first words of each line. Subtitld
sends the subtitles on the clip's span, in the language of the audio. If the
project's subtitles were translated, set the project's source language so
Subtitld can find the original text; otherwise the line is refused with a
message that says so. For the fallback clip in Configure, type its
transcript there too. The add-on guesses the transcript's language from its
words, so its numbers are read in that language.

A good clip is 3 to 10 seconds of one person speaking, clean. Clips under
2 seconds are refused; over 15 seconds tends to give worse takes.

## Options

Set these in **Settings ▸ Add-ons ▸ Reverb ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| CPU threads | automatic | `REVERB_THREADS` | 0 uses the logical processors, at most 8. |
| Quality steps | 8 | `REVERB_STEPS` | The sampler's main passes. 8 is Reverb's own setting; fewer is faster, with more slips. |
| Speed | Exact | `REVERB_PRECISION` | `exact`: the network in full precision, as published (529 MB). `fast`: the network in dynamic int8 (266 MB), about 1.5 to 2 times as fast. Its lines usually sound the same, but it strays further from the published model's predictions. |
| Fallback reference clip | none | `REVERB_VOICE_REF_AUDIO` | Cloned when Subtitld has no clip of the speaker to send. |
| Fallback clip's transcript | none | `REVERB_VOICE_REF_TEXT` | Exactly what is said in the fallback clip. |

Each line is spoken with the seed Subtitld sends, so the same line in the same
voice comes out the same every time. The request's `rate` (−100 to 100 %)
scales the take's length, 0.5 to 2 times the voice's own pace.

## How it runs

Reverb is a 297M-parameter Qwen3 backbone over the codes of the Higgs Audio
v2 codec (25 frames a second, 8 codebooks). It does not generate frame by
frame: the prompt and the reference run once, then 8 passes over all of the
take's frames commit codebooks 0-1, and 2 more fill codebooks 2-7.

`tools/export_models.py` turns Reverb's weights into the files the add-on
downloads, and checks each against Reverb's own package:

- the backbone, re-written with an explicit KV cache so one graph serves the
  prefix and the passes (`backbone.onnx`, weights stored as fp16 and computed
  in fp32, and `backbone_int8.onnx`);
- the codec's encoder and decoder (`codec_encoder.onnx`, `codec_decoder.onnx`);
- the embeddings and heads (`tables.npz`), which numpy applies.

The sampler, the text preparation, the length estimate and the audio helpers
are Reverb's (Apache-2.0), ported to numpy in `reverb_addon/`. The
resampler is torchaudio's, in numpy (`tests/test_audio.py` checks it against
torchaudio). The sampler draws its noise from numpy, not torch, so a seed
gives a different take than Reverb's package does. `tests/test_engine.py`
checks that the Exact network predicts what the package predicts.

## What is bundled, what is downloaded

| What | Size | Licence | Source |
|------|------|---------|--------|
| Add-on, ONNX Runtime, numpy, tokenizers, Verbalis (bundled) | see the release | Apache-2.0; MIT; BSD-3-Clause | this repository, PyPI |
| Codec decoder, tables, tokenizer (downloaded) | 111 MB | **CC BY-NC 4.0** | this repository's [`models-1` release](https://github.com/Subtitld/addon-reverb/releases/tag/models-1) |
| Network, Exact (downloaded) | 529 MB | **CC BY-NC 4.0** | the same |
| Network, Fast (downloaded if chosen) | 266 MB | **CC BY-NC 4.0** | the same |
| Codec encoder, for cloning (downloaded) | 327 MB | **CC BY-NC 4.0** | the same |
| 16 preset voices and the model card (downloaded) | 0.2 MB | **CC BY-NC 4.0** | [software-mansion/Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS) at v1.0.0 |

The files go to Subtitld's model folder (`$SUBTITLD_MODELS_DIR/reverb`), at
the revisions in `reverb_addon/pins.json`, each checked by its sha256. An
interrupted download resumes. `$REVERB_MODELS_URL` and `$HF_ENDPOINT` point
the downloads at mirrors. `MODEL-LICENSE.md` downloads with the model and
says what was changed.

**Use it responsibly.** Clone only voices you have the right to use. Get the
speaker's consent, don't impersonate real people, and don't use generated
speech to deceive, defraud, or harass. Tell your listeners when they hear a
synthetic voice.

Each release archive carries `THIRD_PARTY_LICENSES/README.md`, which names
the notice for every native file in it; the build fails if one has none.
tokenizers' Rust crates are listed with their licences in its software bill
of materials, `THIRD_PARTY_LICENSES/tokenizers/sboms/`. The Linux archive
carries libquadmath (LGPL-2.1), from numpy's wheel; its source is attached to
the same release.

## Limits

- **Non-commercial use only**, because of the model's licence.
- **Seven languages.** A project in any other language is refused with a
  message that says so.
- **Cloning needs the transcript** of the clip (see above).
- **No style control.** Delivery comes from the voice, or the clip.
- **Platforms.** Releases ship Linux x86_64 (glibc 2.35 or newer), macOS arm64
  and Windows x86_64.
- **One line at a time.** A cancel stops the line between sampler passes, or
  stops the download, which resumes on the next line.

## Build

```bash
pip install -c constraints.txt -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/reverb-addon/reverb-addon --self-test
REVERB_TEST_BINARY=$PWD/dist/reverb-addon/reverb-addon pytest tests/test_installed.py
```

`pytest` downloads the model (1.2 GB with both networks and the encoder) into
`$SUBTITLD_MODELS_DIR`.

To make new model files, run `tools/export_models.py` in a separate venv with
torch (its docstring says how), upload its output to a new `models-<n>`
release, set `MODELS_TAG` in `tools/pin_models.py` and run it on the output.

A tag `vYY.MM.N` builds, tests and publishes every platform
(`.github/workflows/release.yml`).
