# Reverb-TTS model files for the Subtitld add-on

These files are converted from **Reverb-TTS v1.0.0** by Software Mansion AI Labs
(https://huggingface.co/software-mansion/Reverb-TTS, revision v1.0.0), with
`tools/export_models.py` of https://github.com/Subtitld/addon-reverb:

- `backbone_int8.onnx`: Reverb's network (initialized from OmniVoice by the k2-fsa team), re-exported with an explicit
  KV cache, its weights quantized to int8.
- `tables.npz`: Reverb's text and audio embeddings and audio heads, as fp16.
- `codec_encoder.onnx`, `codec_decoder.onnx`: the Higgs Audio v2 tokenizer (Boson AI,
  https://huggingface.co/eustlb/higgs-audio-v2-tokenizer), with large weights stored as fp16.
- `tokenizer.json`: Reverb's tokenizer, unchanged.

## Licence

The Reverb weights, and these files made from them, are licensed under
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

You may share and adapt them for **non-commercial purposes only**, giving appropriate credit
(Reverb-TTS by Software Mansion AI Labs), providing a link to the licence and indicating that
changes were made (converted to ONNX, quantized, stored at reduced precision, as listed above).

Reverb's code is Apache-2.0 (https://github.com/software-mansion-labs/Reverb-TTS); the add-on's
sampler and text helpers are ported from it.

## Use it responsibly

Clone only voices you have the right to use. Get the speaker's consent, don't impersonate real
people, and don't use generated speech to deceive, defraud, or harass. Tell your listeners when
they hear a synthetic voice.
