"""Isolated MIT-licensed RawNetLite candidate; never selected by the product.

Unmodified author architecture: lab/vendor/rawnetlite_model.py, Andrea Di Pierno,
2025. License: lab/vendor/RAWNETLITE-LICENSE. Acquisition pins all source/weights.
"""
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from lab.research_downloads import verify

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'fcb0fe8d0f40146d110e322c00a0723ba27d5520'


def preprocess(samples):
    """Author's mono16k path: peak normalize, first 3 s, right zero pad.

    The caller already supplies mono16k. Exact silence remains zero instead of
    the author's undefined zero/zero. No resampling or normalization by label.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite mono16k audio')
    import torch
    waveform = torch.from_numpy(x.copy())[None]
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform/peak
    if waveform.shape[1] < 48000:
        waveform = torch.nn.functional.pad(waveform, (0, 48000-waveform.shape[1]))
    return waveform[:, :48000]


class RawNetLiteDetector:
    threshold = .5
    preprocessing = {'sample_rate': 16000, 'samples': 48000,
                     'normalization': 'peak on entire supplied screen crop, before trimming',
                     'crop': 'first 3 seconds of supplied center-at-most-4-second crop',
                     'padding': 'right zero pad to 3 seconds',
                     'silence': 'return finite zero input instead of upstream zero/zero',
                     'score_direction': 'sigmoid output; native label 1=fake, 0=real'}

    def __init__(self, directory=None):
        directory = Path(directory) if directory else ROOT/'data/research-tools/rawnetlite'/REVISION
        spec = json.loads((ROOT/'docs/rawnetlite-acquisition-v1.json').read_text())
        if spec['revision'] != REVISION:
            raise ValueError('RawNetLite revision mismatch')
        files = {f['name']: f for f in spec['files']}
        checkpoint = 'models/augmented_triple_cross_domain_focal_rawnet_lite.pt'
        # Authenticate before importing source or deserializing any checkpoint.
        manifest = {}
        for name, path in ((checkpoint, directory/checkpoint),
                           ('RawNetLite.py', ROOT/'lab/vendor/rawnetlite_model.py'),
                           ('LICENSE', ROOT/'lab/vendor/RAWNETLITE-LICENSE')):
            entry = files[name]
            if path.stat().st_size != entry['size']:
                raise ValueError('RawNetLite file size mismatch: '+name)
            manifest[name] = verify(path.read_bytes(), maximum=entry['size'],
                                    expected_size=entry['size'], git_blob_sha1=entry['git_blob_sha1'])
        import torch
        from lab.vendor.rawnetlite_model import RawNetLite
        self.torch = torch
        self.model = RawNetLite()
        state = torch.load(directory/checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.manifest = {'repository': spec['repository'], 'revision': REVISION,
                         'license': spec['license'], 'files': manifest,
                         'preprocessing': self.preprocessing, 'native_threshold': .5}
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.timings = []

    def score(self, samples):
        started = time.perf_counter()
        waveform = preprocess(samples)
        with self.torch.inference_mode():
            score = float(self.model(waveform[None]).item())
        elapsed = time.perf_counter()-started
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('RawNetLite returned invalid score')
        self.timings.append({'seconds': elapsed, 'input_seconds': len(samples)/16000,
                             'model_seconds': 3.0})
        return {'synthetic_score': score, 'human_score': 1-score,
                'model_input_sha256': sha256(waveform.numpy().astype('<f4').tobytes()).hexdigest()}
