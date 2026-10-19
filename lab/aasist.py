"""Pinned official AASIST implementation, CPU inference, restricted weight loading."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a04c9863f63d44471dde8a6abcb3b082b07cd1d1'
FILES = {
    'lab/vendor/aasist_model.py': '9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50',
    'data/models/aasist/config.json': 'c25023331685027cce90e1b9a0d2df10aa04b2a27d9b27d5afa36e6815b0fe76',
    'data/models/aasist/model.pth': '51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0',
}


def repeat_pad(x, samples=64600):
    if not len(x) or not np.isfinite(x).all() or len(x) > samples:
        raise ValueError('AASIST window must be finite and contain 1–64600 samples.')
    # Matches the authors' data_utils.pad for short inputs. No truncation here.
    return np.tile(x, int(np.ceil(samples/len(x))))[:samples].astype(np.float32)


class AASISTDetector:
    name = 'AASIST'
    description = 'Spectro-temporal graph attention · official NAVER checkpoint'
    window_samples = 64600
    merge_short_tail = False
    preprocessing = {'normalization': 'none', 'padding': 'repeat waveform to 64600 samples in each short final window, as upstream pad()',
                     'window_tail': 'short tails use a full context window anchored at the file end; overlaps are retained',
                     'aggregation': 'arithmetic mean of contextual window scores; overlapping final windows are not independent',
                     'window_samples': 64600, 'native_class_mapping': {'0': 'spoof', '1': 'bonafide'},
                     'reported_logit_order': ['human / native 1', 'synthetic / native 0']}
    sources = [f'https://github.com/clovaai/aasist/tree/{REVISION}', 'https://arxiv.org/abs/2110.01200']
    class_mapping = {'0': 'human / native class 1 (bonafide)', '1': 'synthetic / native class 0 (spoof)'}
    training_domain = ('Official AASIST checkpoint for ASVspoof 2019 Logical Access (2019-era TTS and voice conversion). '
                       'Vendored MIT-licensed architecture; restricted weights-only loading.')
    limitations = ['Trained on 2019-era attacks. On an independent benchmark of about 25 modern TTS systems it scored near chance (48%).']

    def __init__(self):
        import torch
        manifest = {'repository': 'clovaai/aasist', 'revision': REVISION, 'files': {}}
        for relative, expected in FILES.items():
            p = ROOT/relative
            if not p.exists():
                raise ValueError('AASIST is not installed. Run .venv/bin/python scripts/setup_aasist.py')
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected: raise ValueError('AASIST checksum mismatch: '+relative)
            manifest['files'][relative] = {'sha256': actual, 'bytes': p.stat().st_size}
        from .vendor.aasist_model import Model
        torch.set_num_threads(min(4, max(1, __import__('os').cpu_count() or 1)))
        self.torch = torch
        config = json.loads((ROOT/'data/models/aasist/config.json').read_text())
        self.model = Model(config['model_config'])
        # Restricted tensor/state-dict loader; no unrestricted pickle loading.
        weights = torch.load(ROOT/'data/models/aasist/model.pth', map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.manifest = manifest
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def spans(self, x):
        from .forensics import anchored_spans
        return anchored_spans(len(x), self.window_samples)

    def score(self, x):
        x = repeat_pad(x)
        with self.torch.inference_mode():
            _, output = self.model(self.torch.from_numpy(x).unsqueeze(0), Freq_aug=False)
            logits = output[0][[1, 0]]  # canonical human, synthetic
            p = self.torch.softmax(logits, dim=-1)
        if not self.torch.isfinite(logits).all(): raise ValueError('AASIST returned nonfinite logits.')
        return {'synthetic_score': float(p[1]), 'human_score': float(p[0]), 'logits': [float(v) for v in logits]}
