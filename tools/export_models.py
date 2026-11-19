#!/usr/bin/env python3
"""Make the add-on's model files from Reverb v1.0.0: ONNX graphs and numpy tables.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Run once per model release, in a separate venv with torch (never in the
add-on's own):

    uv venv -p 3.12 /tmp/reverb-export
    VIRTUAL_ENV=/tmp/reverb-export uv pip install --index-url https://download.pytorch.org/whl/cpu \\
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \\
        reverb-tts==1.0.0 onnx onnxruntime
    /tmp/reverb-export/bin/python tools/export_models.py models-out

It writes, and checks each against the package:

- backbone.onnx: the backbone re-written with an explicit KV cache (one
  graph for the prefix and for the passes), its weights stored as fp16 and
  cast back to fp32: the package's maths, half the download.
- backbone_int8.onnx: the same graph with dynamic int8 quantization, about
  twice as fast on a CPU but further from the package (Configure's "Fast").
- codec_decoder.onnx, codec_encoder.onnx: the codec, with large weights
  stored as fp16 and cast back to fp32 inside the graph.
- tables.npz: embeddings and heads, as fp16.
- tokenizer.json: the model's tokenizer.
- MODEL-LICENSE.md: what the files are, and their licence (CC BY-NC 4.0).

Then tools/pin_models.py records their sizes and sha256 in pins.json, and the
files are uploaded to the `models-<n>` release of this repository.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from reverb_tts import TTS
from reverb_tts.model import CODES, MASK
from reverb_tts.sampler import Batch, Prompt
from reverb_tts.tts import REPOSITORY, REVISION, _prompt

OPSET = 17


def backbone_modules(llm):
    cfg = llm.config
    L, NH, KVH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    theta = cfg.rope_parameters['rope_theta'] if getattr(cfg, 'rope_parameters', None) else cfg.rope_theta
    eps = cfg.rms_norm_eps
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    assert (L, KVH, D) == (24, 6, 128), 'engine.py assumes 24 layers of 6 KV heads of 128'

    def rms(x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    def rope(x, pos):
        f = pos[:, None, :, None].float() * inv[None, None, None, :]
        emb = torch.cat([f, f], -1)
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        return x * emb.cos() + torch.cat([-x2, x1], -1) * emb.sin()

    class Backbone(torch.nn.Module):
        """Qwen3 decoder layers with the KV cache as plain tensors: [layers, batch, kv heads, length, dim]."""

        def __init__(self):
            super().__init__()
            self.layers, self.norm = llm.layers, llm.norm

        def forward(self, x, mask, pos, past_k, past_v):
            B, T, _ = x.shape
            ks, vs = [], []
            for i, layer in enumerate(self.layers):
                a = layer.self_attn
                h = rms(x, layer.input_layernorm.weight)
                q = a.q_proj(h).view(B, T, NH, D).transpose(1, 2)
                k = a.k_proj(h).view(B, T, KVH, D).transpose(1, 2)
                v = a.v_proj(h).view(B, T, KVH, D).transpose(1, 2)
                q, k = rope(rms(q, a.q_norm.weight), pos), rope(rms(k, a.k_norm.weight), pos)
                ks.append(k)
                vs.append(v)
                k, v = torch.cat([past_k[i], k], 2), torch.cat([past_v[i], v], 2)
                k, v = k.repeat_interleave(NH // KVH, 1), v.repeat_interleave(NH // KVH, 1)
                att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D) + mask, -1) @ v
                x = x + a.o_proj(att.transpose(1, 2).reshape(B, T, NH * D))
                m = layer.mlp
                h = rms(x, layer.post_attention_layernorm.weight)
                x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
            return rms(x, self.norm.weight), torch.stack(ks), torch.stack(vs)

    return Backbone().eval(), (L, KVH, D)


def store_fp16(src: Path, dst: Path) -> None:
    """Large float weights stored as fp16, each cast back to fp32 by the graph: half the file, fp32 maths."""
    model = onnx.load(str(src))
    graph = model.graph
    # The TorchScript exporter leaves some weights as Constant nodes: make them initializers first.
    for node in list(graph.node):
        if node.op_type == 'Constant' and node.attribute[0].name == 'value':
            tensor = node.attribute[0].t
            if tensor.data_type == TensorProto.FLOAT and numpy_helper.to_array(tensor).size >= 1024:
                graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(tensor), node.output[0]))
                graph.node.remove(node)
    casts = []
    for init in graph.initializer:
        array = numpy_helper.to_array(init) if init.data_type == TensorProto.FLOAT else None
        # A tensor past fp16's range (the decoder has one, at 2.4e6) stays fp32.
        if array is not None and array.size >= 1024 and float(np.abs(array).max()) < 6e4:
            name = init.name
            init.CopyFrom(numpy_helper.from_array(array.astype(np.float16), name + '__fp16'))
            casts.append(helper.make_node('Cast', [name + '__fp16'], [name], to=TensorProto.FLOAT))
    for cast in reversed(casts):
        graph.node.insert(0, cast)
    onnx.save(model, str(dst))


def session(path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / 'scratch'
    scratch.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    tts = TTS.from_pretrained(variant='int8', device='cpu')
    net, codec = tts.network.requires_grad_(False), tts.codec.requires_grad_(False).eval()
    backbone, (L, KVH, D) = backbone_modules(net.llm)
    voice = tts.preset_voice('en_male_narrator_1')
    offsets = CODES * torch.arange(8)[:, None]

    with torch.inference_mode():
        # The backbone, checked against the package's own pass.
        ids = tts.tokenizer(_prompt('Hello there, this is a test.', voice.text, 'en'), return_tensors='pt').input_ids[0]
        batch = Batch(net, [Prompt(ids, voice.codes, 60, 1)], 1.0)
        tokens = torch.full((1, 8, 60), MASK)
        tokens[:, :2, ::3] = torch.randint(0, 1024, (1, 2, 20))
        reference = batch.log_probs(tokens, range(0, 8), polish=True)
        prefix = torch.cat([net.llm.embed_tokens(ids), net.audio_embeddings(voice.codes + offsets).sum(0)])[None]
        P = prefix.size(1)
        empty = torch.zeros(L, 1, KVH, 0, D)
        _, k, v = backbone(prefix, torch.zeros(1, 1, P, P), torch.arange(P)[None], empty, empty)
        x = net.audio_embeddings(tokens + CODES * torch.arange(8)[None, :, None]).sum(1)
        h, _, _ = backbone(x, batch.masks[True], batch.positions, k, v)
        lp = (h @ net.audio_heads.weight.T).view(1, 60, 8, CODES).transpose(1, 2).log_softmax(-1)[..., :MASK]
        error = float((lp - reference).abs().max())
        print(f'backbone port: max |d log p| = {error:.2e}')
        assert error < 5e-3

        torch.onnx.export(backbone, (x, batch.masks[True], batch.positions, k, v), scratch / 'backbone.onnx',
                          input_names=['embeds', 'mask', 'positions', 'past_keys', 'past_values'],
                          output_names=['hidden', 'keys', 'values'], dynamo=False, opset_version=OPSET,
                          dynamic_axes={'embeds': {0: 'B', 1: 'T'}, 'mask': {0: 'B', 2: 'T', 3: 'PT'},
                                        'positions': {0: 'B', 1: 'T'}, 'past_keys': {1: 'B', 3: 'P'},
                                        'past_values': {1: 'B', 3: 'P'}, 'hidden': {0: 'B', 1: 'T'},
                                        'keys': {1: 'B', 3: 'T'}, 'values': {1: 'B', 3: 'T'}})
        store_fp16(scratch / 'backbone.onnx', out / 'backbone.onnx')
        quantize_dynamic(str(scratch / 'backbone.onnx'), str(out / 'backbone_int8.onnx'),
                         weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'], per_channel=True)

        embed = net.llm.embed_tokens
        np.savez(out / 'tables.npz', text_table=embed.table.weight.numpy().astype(np.float16),
                 text_project=embed.project.weight.numpy().astype(np.float16),
                 audio_embeddings=net.audio_embeddings.weight.numpy().astype(np.float16),
                 audio_heads=net.audio_heads.weight.numpy().astype(np.float16))

        class Decoder(torch.nn.Module):
            def forward(self, codes):
                return codec.decode(codes).audio_values

        torch.onnx.export(Decoder(), (voice.codes[None],), scratch / 'codec_decoder.onnx', input_names=['codes'],
                          output_names=['audio'], dynamo=False, opset_version=OPSET,
                          dynamic_axes={'codes': {0: 'B', 2: 'T'}, 'audio': {0: 'B', 2: 'S'}})
        store_fp16(scratch / 'codec_decoder.onnx', out / 'codec_decoder.onnx')
        wav = codec.decode(voice.codes[None]).audio_values.numpy()
        mine = session(out / 'codec_decoder.onnx').run(None, {'codes': voice.codes[None].numpy()})[0]
        snr = 10 * np.log10((wav ** 2).sum() / ((wav - mine) ** 2).sum())
        print(f'codec decoder (fp16 storage): {snr:.1f} dB SNR against the package')
        assert snr > 30

        cfg = codec.config

        class Encoder(torch.nn.Module):
            """24 kHz audio padded to whole frames, and its 16 kHz copy, -> codes [1, 8, frames]."""

            def forward(self, wav24, wav16):
                hidden = codec.semantic_model(F.pad(wav16, (160, 160)), output_hidden_states=True).hidden_states
                semantic = torch.stack(list(hidden), 1).mean(1)[:, :: cfg.semantic_downsample_factor, :]
                e_semantic = codec.encoder_semantic(semantic.transpose(1, 2))
                e_acoustic = codec.acoustic_encoder(wav24)  # whole frames: the lengths already agree
                emb = codec.fc(torch.cat([e_acoustic, e_semantic], 1).transpose(1, 2)).transpose(1, 2)
                return codec.quantizer.encode(emb, None).transpose(0, 1)

        def pair(n: int):
            w = F.pad(torch.randn(n) * 0.1, (0, -n % cfg.hop_length))
            return w[None, None], torchaudio.functional.resample(w, 24000, 16000)[None]

        for n in (24000 * 3 + 5, 24000 * 9 + 1000):
            a, b = pair(n)
            same = float((codec.encode(a).audio_codes == Encoder()(a, b)).float().mean())
            print(f'codec encoder port, {n} samples: {same:.1%} of codes match')
            assert same == 1.0
        a, b = pair(24000 * 5 + 77)
        torch.onnx.export(Encoder(), (a, b), scratch / 'codec_encoder.onnx', input_names=['wav24', 'wav16'],
                          output_names=['codes'], dynamo=False, opset_version=OPSET,
                          dynamic_axes={'wav24': {2: 'N'}, 'wav16': {1: 'M'}, 'codes': {2: 'T'}})
        store_fp16(scratch / 'codec_encoder.onnx', out / 'codec_encoder.onnx')
        a, b = pair(24000 * 7)
        ours = session(out / 'codec_encoder.onnx').run(None, {'wav24': a.numpy(), 'wav16': b.numpy()})[0]
        same = float((codec.encode(a).audio_codes.numpy() == ours).mean())
        print(f'codec encoder (fp16 storage): {same:.1%} of codes match')
        assert same > 0.9

    shutil.copy(hf_hub_download(REPOSITORY, 'model/tokenizer.json', revision=REVISION), out / 'tokenizer.json')
    shutil.copy(Path(__file__).with_name('MODEL-LICENSE.md'), out / 'MODEL-LICENSE.md')
    shutil.rmtree(scratch)
    for path in sorted(out.iterdir()):
        print(f'{path.name}: {path.stat().st_size:,} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else 'models-out')))
