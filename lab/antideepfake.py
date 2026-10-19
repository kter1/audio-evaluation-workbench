"""NII AntiDeepfake MMS-300M, ported from fairseq to transformers. CPU inference, safetensors only.

The official checkpoint stores a fairseq Wav2Vec2Model front end plus mean pooling and a 2-class
linear head (class 0 = fake, class 1 = real). The same architecture exists in transformers; this
module renames the checkpoint's keys and drops the unused pre-training quantizer.
scripts/check_antideepfake_parity.py verifies the port against the official fairseq code.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'data/models/antideepfake'
REPOSITORY = 'nii-yamagishilab/mms-300m-anti-deepfake'
REVISION = '7928e119e09f616b2ce698892e00ca22dbaa40bf'
FILES = {
    'model.safetensors': '9bd5d9785bf72f91dfca8508651354946c4b307921cb358505e1b94035102224',
    'config.json': 'be5a16e7bf332832f0a5583357ed37dd67cc4a39abd24832cf6a74a8217b046c',
    'README.md': 'd7139a3c76480f0167f2e768b7af833659541453d7b45f277e3774e98ad1493f',
}
WINDOW = 64000  # 4 s; matches pellav2 so the two can be fused window by window
UNUSED = ('m_ssl.model.quantizer.', 'm_ssl.model.project_q.', 'm_ssl.model.final_proj.')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def convert(state):
    """fairseq checkpoint keys -> (transformers Wav2Vec2Model state, head state)."""
    backbone, head = {}, {}
    for key, value in state.items():
        if key.startswith(UNUSED):
            continue
        if key.startswith('proj_fc.'):
            head[key[len('proj_fc.'):]] = value
            continue
        if not key.startswith('m_ssl.model.'):
            raise ValueError(f'Unexpected AntiDeepfake checkpoint key: {key}')
        k = key[len('m_ssl.model.'):]
        parts = k.split('.')
        if k == 'mask_emb':
            new = 'masked_spec_embed'
        elif k.startswith('feature_extractor.conv_layers.'):
            i, sub = parts[2], parts[3]
            new = f'feature_extractor.conv_layers.{i}.' + ('conv.' if sub == '0' else 'layer_norm.') + parts[-1]
        elif k.startswith('layer_norm.'):
            new = 'feature_projection.layer_norm.' + parts[-1]
        elif k.startswith('post_extract_proj.'):
            new = 'feature_projection.projection.' + parts[-1]
        elif k.startswith('encoder.pos_conv.0.'):
            new = {'bias': 'encoder.pos_conv_embed.conv.bias',
                   'weight_g': 'encoder.pos_conv_embed.conv.parametrizations.weight.original0',
                   'weight_v': 'encoder.pos_conv_embed.conv.parametrizations.weight.original1'}[parts[-1]]
        elif k.startswith('encoder.layer_norm.'):
            new = 'encoder.layer_norm.' + parts[-1]
        elif k.startswith('encoder.layers.'):
            i, rest = parts[2], '.'.join(parts[3:])
            rest = (rest.replace('self_attn_layer_norm.', 'layer_norm.')
                        .replace('self_attn.', 'attention.')
                        .replace('fc1.', 'feed_forward.intermediate_dense.')
                        .replace('fc2.', 'feed_forward.output_dense.'))
            new = f'encoder.layers.{i}.{rest}'
        else:
            raise ValueError(f'Unmapped AntiDeepfake checkpoint key: {key}')
        backbone[new] = value
    return backbone, head


def prepare(x, samples=WINDOW):
    """Official preprocessing: layer-normalize the whole (mono, 16 kHz) waveform; no padding."""
    x = np.asarray(x, dtype=np.float32)
    if not len(x) or not np.isfinite(x).all() or len(x) > samples:
        raise ValueError(f'AntiDeepfake window must be finite and contain 1–{samples} samples.')
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-5)).astype(np.float32)


class AntiDeepfakeDetector:
    name = 'AntiDeepfake MMS-300M'
    description = 'MMS-300M · post-trained on 56k h real + 18k h artifacted speech · NII'
    window_samples = WINDOW
    threshold = 0.5
    class_mapping = {'0': 'human / native class 1 (real)', '1': 'synthetic / native class 0 (fake)'}
    training_domain = ('Post-trained by NII on about 56,000 hours of real speech and 18,000 hours of synthesized, '
                       'converted, vocoded, and codec speech across 100+ languages, with data augmentation. '
                       'CC BY-NC-SA 4.0 weights: research and educational use only.')
    limitations = ['Non-commercial weights (CC BY-NC-SA 4.0).',
                   'Scores 4 s windows; the authors score whole utterances. Shorter context can lower accuracy.']
    preprocessing = {'normalization': 'per-window layer norm of the waveform (zero mean, unit variance, eps 1e-5), as upstream',
                     'padding': 'none; a window shorter than 4 s is scored at its own length',
                     'window_tail': 'complete 4 s windows; a remaining tail gets a full window anchored at the file end',
                     'aggregation': 'arithmetic mean of window softmax scores',
                     'window_samples': WINDOW,
                     'output': 'native logits [fake, real], reported in human, synthetic order'}
    sources = [f'https://huggingface.co/{REPOSITORY}/tree/{REVISION}', 'https://arxiv.org/abs/2506.21090']

    def __init__(self, directory=MODEL_DIR):
        import torch
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        directory = Path(directory)
        manifest = {'repository': REPOSITORY, 'revision': REVISION, 'files': {}}
        for name, expected in FILES.items():
            path = directory / name
            if not path.exists():
                raise ValueError('AntiDeepfake is not installed. Run .venv/bin/python scripts/setup_antideepfake.py')
            if file_hash(path) != expected:
                raise ValueError('AntiDeepfake checksum mismatch: ' + name)
            manifest['files'][name] = {'sha256': expected, 'bytes': path.stat().st_size}
        torch.set_num_threads(min(4, max(1, __import__('os').cpu_count() or 1)))
        self.torch = torch
        backbone, head = convert(load_file(directory / 'model.safetensors'))
        self.backbone = Wav2Vec2Model(Wav2Vec2Config.from_dict(json.loads((directory / 'config.json').read_text())))
        self.backbone.load_state_dict(backbone, strict=True)
        self.head = torch.nn.Linear(1024, 2)
        self.head.load_state_dict(head, strict=True)
        self.backbone.eval()
        self.head.eval()
        self.manifest = manifest
        self.parameters = sum(p.numel() for p in self.backbone.parameters()) + sum(p.numel() for p in self.head.parameters())

    def spans(self, x):
        from .forensics import anchored_spans
        return anchored_spans(len(x), self.window_samples)

    def logits(self, x, samples=WINDOW):
        """Native [fake, real] logits for one window (or a whole clip when samples allows it)."""
        with self.torch.inference_mode():
            hidden = self.backbone(self.torch.from_numpy(prepare(x, samples))[None]).last_hidden_state
            return self.head(hidden.mean(dim=1))[0]

    def score(self, x):
        native = self.logits(x)
        p = self.torch.softmax(native, dim=-1)
        fake, real = float(native[0]), float(native[1])
        if not np.isfinite([fake, real]).all():
            raise ValueError('AntiDeepfake returned nonfinite logits.')
        return {'synthetic_score': float(p[0]), 'human_score': float(p[1]), 'logits': [real, fake]}
