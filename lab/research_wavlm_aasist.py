"""Pinned research-only WavLM-large/AASIST classifier, with offline construction.

Author classes are authenticated before their definitions are compiled. Every
backbone weight comes from the verified safetensors checkpoint; Hub downloads,
remote initialization, and pickle loading are never used. CC-BY-NC-4.0.
"""
import ast
from hashlib import sha1, sha256
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'eliya/forensics_0.3B_base_deepfake_classifier'
REVISION = 'cfd11365b02167214dc3b392378809fef6ca3cfa'
BACKBONE_REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
WINDOW = 80000
CHECKPOINT = 'checkpoint_epoch_5.safetensors'
CHECKPOINT_BYTES = 1269932956
CHECKPOINT_SHA256 = '803d2c57e705395d007b9deaf5b1a110965950f207df223e6dff5b6d298490fb'
SOURCE_FILES = {
    'README.md': (6155, '669489393b1f590fb87522c90d9a0aab79a57c4a'),
    'model.py': (7763, 'a27455a6ed46ee0a1d9a69ed8f7b6cd4ad7b0ebf'),
    'config.json': (190, 'd35749b23b8ef4665bc0afdc60a4f19be016bf90'),
    'inference.py': (2578, 'bf9b8d6e2c9847480b6d532c35338976cc52a88f'),
    'wavlm-large-config.json': (2222, 'b1d1becf90dd05db908a9114148c204484ebec69'),
}
CLASS_NAMES = {'_GraphAttentionLayer', '_HtrgGraphAttentionLayer', '_GraphPool',
               'AASISTPool', 'MLPBlock', 'Classifier', 'DeepfakeDetector'}
UNUSED_PROJECTION = {
    'projection.0.bias': ((320,), '268a8c4e0c1b39219fb86b7fde69764111efeb36b117cb9ba4acc36b9a5e95dd'),
    'projection.0.weight': ((320, 320), '3752915c4a27fc70ed850ca8e50f8261113eded1c103d64b386ddd3c51a5c2a9'),
    'projection.2.bias': ((320,), '5666395e1eb42d65b9cf15989d921b9464dfb277630de3375967ee4c4333997e'),
    'projection.2.weight': ((320, 320), '9daea865528424cd39519558e77af44f7ce601c01acce1eb6663e8f4684db687'),
}


def verify_source(directory, name):
    path = Path(directory)/name
    size, expected = SOURCE_FILES[name]
    if path.stat().st_size != size:
        raise ValueError('WavLM/AASIST file size mismatch: '+name)
    raw = path.read_bytes()
    if sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() != expected:
        raise ValueError('WavLM/AASIST Git blob identity mismatch: '+name)
    return {'bytes': size, 'sha256': sha256(raw).hexdigest(), 'git_blob_sha1': expected}


def verify_checkpoint(directory):
    path = Path(directory)/CHECKPOINT
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError('WavLM/AASIST checkpoint size mismatch')
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('WavLM/AASIST checkpoint SHA-256 mismatch')
    return {'bytes': CHECKPOINT_BYTES, 'sha256': CHECKPOINT_SHA256}


def validate_state(state):
    import torch
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError('WavLM/AASIST checkpoint contains a non-tensor: '+name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('WavLM/AASIST checkpoint contains nonfinite weights: '+name)


def inference_state(state, model_keys):
    """Allow only four authenticated parameters absent from published inference.

    The author's inference uses strict=False. Its model.py never creates or
    references `projection`; the published forward uses pool/classifier directly.
    Keep strict loading for every inference parameter and reject any other extra.
    """
    extras = set(state)-set(model_keys)
    if extras != set(UNUSED_PROJECTION):
        raise ValueError('Unexpected WavLM/AASIST checkpoint keys: '+', '.join(sorted(extras)))
    dropped = []
    for name, (shape, expected_hash) in UNUSED_PROJECTION.items():
        value = state[name]
        if tuple(value.shape) != shape or str(value.dtype) != 'torch.float32':
            raise ValueError('Unused WavLM/AASIST projection shape/dtype mismatch: '+name)
        digest = sha256(value.numpy().astype('<f4').tobytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError('Unused WavLM/AASIST projection SHA-256 mismatch: '+name)
        dropped.append({'name': name, 'shape': list(shape), 'dtype': str(value.dtype),
                        'sha256': digest,
                        'reason': 'Not constructed or referenced by pinned published inference architecture'})
    return {name: value for name, value in state.items() if name not in extras}, dropped


def preprocess(samples):
    """Exact author mono16k tensor path: peak normalize, repeat/center to 5 s."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite mono16k audio')
    import torch
    waveform = torch.from_numpy(x.copy())
    waveform = waveform/(waveform.abs().max()+1e-8)
    current = waveform.shape[0]
    if current < WINDOW:
        waveform = waveform.repeat((WINDOW+current-1)//current)[:WINDOW]
    elif current > WINDOW:
        start = (current-WINDOW)//2
        waveform = waveform[start:start+WINDOW]
    return waveform[None]


def author_classes(directory):
    verify_source(directory, 'model.py')
    import torch
    tree = ast.parse((Path(directory)/'model.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in CLASS_NAMES]
    if {n.name for n in definitions} != CLASS_NAMES:
        raise ValueError('Pinned WavLM/AASIST classes are incomplete')
    detector = next(n for n in definitions if n.name == 'DeepfakeDetector')
    # Remove only the constructor which downloads pretrained backbone weights.
    detector.body = [n for n in detector.body
                     if not (isinstance(n, ast.FunctionDef) and n.name == '__init__')]
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'pinned-wavlm-aasist-model', 'exec'), namespace)
    return {name: namespace[name] for name in CLASS_NAMES}


def build_model(directory, config):
    import torch
    from transformers import WavLMConfig, WavLMModel
    classes = author_classes(directory)

    class OfflineDetector(classes['DeepfakeDetector']):
        def __init__(self):
            super().__init__()
            self.wavlm = WavLMModel(WavLMConfig.from_dict(config))
            self.pool = classes['AASISTPool'](
                self.wavlm.config.hidden_size, self.wavlm.config.num_hidden_layers, (128, 64))
            self.classifier = classes['Classifier'](self.pool.output_dim, 320, 3)

    return OfflineDetector()


class WavLMAASISTDetector:
    name = 'WavLM-large + AASIST (research candidate)'
    description = 'Fully fine-tuned WavLM-large with AASIST pooling; fixed publisher operating point'
    threshold = .5
    window_samples = WINDOW
    class_mapping = {'0': 'human / native label1=real',
                     '1': 'synthetic / 1-sigmoid(native real logit)'}
    training_domain = 'Publisher reports multi-source genuine and synthetic speech with codec and noise augmentation; detailed training overlap is unknown.'
    limitations = [
        'CC-BY-NC-4.0 license limits commercial use.',
        'Training-data overlap with screening sources and generators is unknown.',
        'The fixed 0.5 inference threshold has not been calibrated to this product or target population.',
        'Raw scores are not established authenticity probabilities.',
    ]
    sources = [f'https://huggingface.co/{REPOSITORY}/tree/{REVISION}',
               f'https://huggingface.co/microsoft/wavlm-large/tree/{BACKBONE_REVISION}']
    preprocessing = {
        'sample_rate': 16000, 'samples': WINDOW,
        'normalization': 'divide entire supplied mono16k crop by its peak + 1e-8',
        'padding': 'repeat short waveform and trim to 5 seconds',
        'crop': 'center 5 seconds if supplied input is longer',
        'score_direction': 'native label1=real; fake score=1-sigmoid(native real logit)',
        'threshold': 'author inference.py fixed fake score >=0.5; no threshold tuning',
    }

    def __init__(self, directory=None):
        directory = Path(directory) if directory else ROOT/'data/research-tools/wavlm-aasist'/REVISION
        # Verify before importing model dependencies, source compilation or loading.
        checkpoint = verify_checkpoint(directory)
        sources = {name: verify_source(directory, name) for name in SOURCE_FILES}
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        state = load_file(directory/CHECKPOINT, device='cpu')
        validate_state(state)
        config = json.loads((directory/'wavlm-large-config.json').read_text())
        self.model = build_model(directory, config)
        filtered_state, dropped = inference_state(state, self.model.state_dict())
        self.model.load_state_dict(filtered_state, strict=True)
        del filtered_state
        del state
        self.model.eval()
        self.manifest = {
            'repository': REPOSITORY, 'revision': REVISION, 'license': 'CC-BY-NC-4.0',
            'backbone': 'microsoft/wavlm-large', 'backbone_revision': BACKBONE_REVISION,
            'files': {CHECKPOINT: checkpoint, **sources}, 'preprocessing': self.preprocessing,
            'native_threshold': self.threshold,
            'unused_checkpoint_parameters': dropped,
        }
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.timings = []

    def spans(self, samples):
        from .forensics import anchored_spans
        return anchored_spans(len(samples), self.window_samples)

    def score(self, samples):
        started = time.perf_counter()
        waveform = preprocess(samples)
        with self.torch.inference_mode():
            native = self.model(waveform)[0]
            real_score = float(self.torch.sigmoid(native))
        native_logit = float(native)
        score = 1.-real_score
        if not np.isfinite(native_logit) or not 0 <= score <= 1:
            raise ValueError('WavLM/AASIST returned an invalid score')
        self.timings.append({'seconds': time.perf_counter()-started,
                             'input_seconds': len(samples)/16000, 'model_seconds': 5.})
        return {'synthetic_score': score, 'human_score': real_score,
                'logits': [0., -native_logit], 'native_real_logit': native_logit,
                'model_input_sha256': sha256(waveform.numpy().astype('<f4').tobytes()).hexdigest()}
