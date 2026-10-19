"""Pinned pellav2 detector: XLS-R-300M with learned layer weighting, CPU inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'data/models/pellav2'
REPOSITORY = 'Sadanie/pellav2-audio-deepfake-detector'
REVISION = 'a4c68a3d4afc7f5d9450c94f5d0652aaa208d40a'
BACKBONE = 'facebook/wav2vec2-xls-r-300m'
BACKBONE_REVISION = '1a640f32ac3e39899438a2931f9924c02f080a54'
# Published path -> (local name, SHA-256). Upstream files are fixed by the pinned revisions.
FILES = {
    f'{REPOSITORY}/resolve/{REVISION}/pellav2_detector.pt':
        ('pellav2_detector.pt', '8214705ad784f66bca0736300e3432d5bfc325f1fc211a3aea5f9a13c790b023'),
    f'{REPOSITORY}/resolve/{REVISION}/pellav2_infer.py':
        ('pellav2_infer.py', '2ebee7a028a64ed0bf263ec282ad45d8bde84044293394aafb6b0c26fb0f31e6'),
    f'{REPOSITORY}/resolve/{REVISION}/config.json':
        ('config.json', '5a0fe2addbd42f4964d8e4fc4f2c8e70f817db84992094fff926ee1e6933e4a5'),
    f'{BACKBONE}/resolve/{BACKBONE_REVISION}/config.json':
        ('xls-r-300m-config.json', '0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac'),
}
WINDOW = 64000  # 4 s at 16 kHz, the upstream crop length


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(x, samples=WINDOW):
    """Upstream load_clip order: zero-pad the end to the crop length, then standardize."""
    x = np.asarray(x, dtype=np.float32)
    if not len(x) or not np.isfinite(x).all() or len(x) > samples:
        raise ValueError(f'pellav2 window must be finite and contain 1–{samples} samples.')
    x = np.pad(x, (0, samples-len(x)))
    return ((x-x.mean()) / (x.std()+1e-7)).astype(np.float32)


class PellaV2Detector:
    name = 'pellav2'
    description = 'XLS-R-300M · learned layer weighting · modern TTS training'
    window_samples = WINDOW
    threshold = 0.5
    class_mapping = {'0': 'human / fixed reference logit 0', '1': 'synthetic / model logit; softmax([0, l]) = sigmoid(l)'}
    training_domain = ('Curated private dataset of real speech and synthetic speech from a broad range of modern TTS systems '
                       '(per the model card; details unpublished). MIT license. Scores calibrated by the authors for a 0.5 threshold.')
    limitations = ['Training data is private and may include the generators used in this lab’s modern-voice test; '
                   'results there are fresh clips from popular generators, not unseen generator families.',
                   'Noisy real-world channels (interviews, music beds, heavy recompression) raise false positives on genuine speech, per the model card.',
                   'Voice conversion, singing, and non-English speech are not evaluated by the authors.']
    preprocessing = {'normalization': 'per-window (x - mean) / (std + 1e-7) after zero padding, as upstream load_clip',
                     'padding': 'zero-pad the end of a window shorter than 64000 samples, as upstream',
                     'window_tail': 'complete 4 s windows; a remaining tail gets a full window anchored at the file end',
                     'aggregation': 'arithmetic mean of window sigmoid scores; upstream scores one centre 4 s crop instead',
                     'window_samples': WINDOW,
                     'output': 'one logit l; synthetic score = sigmoid(l); reported logits [0, l]'}
    sources = [f'https://huggingface.co/{REPOSITORY}/tree/{REVISION}',
               f'https://huggingface.co/{BACKBONE}/tree/{BACKBONE_REVISION}']

    def __init__(self, directory=MODEL_DIR):
        import torch
        directory = Path(directory)
        manifest = {'repository': REPOSITORY, 'revision': REVISION,
                    'backbone': BACKBONE, 'backbone_revision': BACKBONE_REVISION, 'files': {}}
        for local, expected in FILES.values():
            path = directory/local
            if not path.exists():
                raise ValueError('pellav2 is not installed. Run .venv/bin/python scripts/setup_pellav2.py')
            if file_hash(path) != expected:
                raise ValueError('pellav2 checksum mismatch: '+local)
            manifest['files'][local] = {'sha256': expected, 'bytes': path.stat().st_size}
        from .vendor.pellav2_model import Detector
        torch.set_num_threads(min(4, max(1, __import__('os').cpu_count() or 1)))
        self.torch = torch
        self.model = Detector(json.loads((directory/'xls-r-300m-config.json').read_text()))
        # Restricted tensor/state-dict loader; no unrestricted pickle loading.
        weights = torch.load(directory/'pellav2_detector.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.manifest = manifest
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def spans(self, x):
        from .forensics import anchored_spans
        return anchored_spans(len(x), self.window_samples)

    def score(self, x):
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(prepare(x))[None])[0]
            p = float(self.torch.sigmoid(output))
        logit = float(output)
        if not np.isfinite(logit):
            raise ValueError('pellav2 returned a nonfinite logit.')
        return {'synthetic_score': p, 'human_score': 1-p, 'logits': [0.0, logit]}
