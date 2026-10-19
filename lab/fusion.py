"""Score-level fusion of pellav2 and AntiDeepfake: a two-input logistic model per 4 s window.

Base: fused logit = b0 + b1·a + b2·b, where a = pellav2 logit and b = AntiDeepfake (fake − real) logit.
Noise-aware: adds n, a·n, b·n, where n is a noise index measured from the window's own audio, so the
fusion can weigh the two models differently when noise fills the pauses in speech.
Weights and the decision threshold are frozen in lab/fusion-v4.json by scripts/fit_fusion.py from
development data only. The file score is the mean of window probabilities, as for pellav2.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'lab/fusion-v4.json'
CLIP = 30.0  # bound extreme logits so one saturated model cannot dominate numerically


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def noise_index(x, frame=320):
    """0 for speech with clean pauses, rising to 2 as noise fills them.

    D = 95th minus 10th percentile of 20 ms frame RMS in dB; n = clip((40 - D) / 20, 0, 2).
    """
    x = np.asarray(x, dtype=np.float64)
    frames = x[:len(x) // frame * frame].reshape(-1, frame)
    if not len(frames):
        return 0.0
    db = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), 1e-5))
    spread = np.percentile(db, 95) - np.percentile(db, 10)
    return float(np.clip((40 - spread) / 20, 0, 2))


def design(features):
    """Rows of [a, b] (base) or [a, b, n] (noise-aware) -> design matrix with bias and interactions."""
    features = np.asarray(features, dtype=np.float64)
    features = features.reshape(-1, features.shape[-1] if features.ndim > 1 else len(features))
    logits = np.clip(features[:, :2], -CLIP, CLIP)
    columns = [np.ones((len(features), 1)), logits]
    if features.shape[1] == 3:
        n = features[:, 2:3]
        columns += [n, logits * n]
    return np.hstack(columns)


def fit_logistic(features, labels, sample_weight=None, l2=1.0, iterations=100):
    """Weighted, L2-penalized logistic regression by Newton iterations; the bias is not penalized."""
    A, y = design(features), np.asarray(labels, dtype=np.float64)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    w = w * len(w) / w.sum()
    penalty = np.full(A.shape[1], float(l2))
    penalty[0] = 0.0
    beta = np.zeros(A.shape[1])
    for _ in range(iterations):
        p = sigmoid(A @ beta)
        gradient = A.T @ (w * (p - y)) + penalty * beta
        hessian = (A * (w * p * (1 - p))[:, None]).T @ A + np.diag(penalty) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def window_probabilities(beta, features):
    return sigmoid(design(features) @ np.asarray(beta, dtype=np.float64))


def file_score(beta, features):
    """Mean of window probabilities: the same aggregation the lab uses for single detectors."""
    return float(np.mean(window_probabilities(beta, features)))


def load_spec(path=WEIGHTS):
    text = Path(path).read_text()
    return json.loads(text), hashlib.sha256(text.encode()).hexdigest()


class FusionDetector:
    name = 'pellav2 + AntiDeepfake fusion'
    description = 'Two 300M speech models · logistic fusion fitted on development data · protocol v4'
    band_logit = 2.0  # abstention band: threshold ± 2 on the fused-logit scale
    class_mapping = {'0': 'human / fixed reference logit 0', '1': 'synthetic / fused logit; softmax([0, z]) = sigmoid(z)'}
    training_domain = ('pellav2 (private modern-TTS training, MIT) and NII AntiDeepfake MMS-300M (56k h real + 18k h '
                       'artifacted speech, CC BY-NC-SA 4.0), combined by a 3-parameter logistic model fitted on the '
                       'already-reported v3 clips in clean, MP3, telephone, and noisy versions. Non-commercial overall.')
    limitations = ['Combined detector inherits AntiDeepfake’s non-commercial license (CC BY-NC-SA 4.0).',
                   'Fusion weights and threshold were fitted to Chatterbox-Turbo and F5-TTS clips; XTTS-v2 is held out to test generalization, but other generators may score differently.']
    sources = ['https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector',
               'https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake', 'https://arxiv.org/abs/2506.21090']

    def __init__(self, weights=WEIGHTS):
        from .antideepfake import AntiDeepfakeDetector
        from .pella import PellaV2Detector
        if not Path(weights).exists():
            raise ValueError('Fusion weights are not frozen yet. Run scripts/fit_fusion.py')
        spec, sha = load_spec(weights)
        self.pella, self.adf = PellaV2Detector(), AntiDeepfakeDetector()
        if spec['models'] != {'pellav2': self.pella.manifest, 'antideepfake': self.adf.manifest}:
            raise ValueError('Fusion weights were fitted to different detector files')
        self.beta = np.array(spec['beta'], dtype=np.float64)
        self.noise_aware = spec.get('kind') == 'noise-aware'
        self.threshold = float(spec['threshold'])
        self.window_samples = self.pella.window_samples
        self.manifest = {'repository': 'pellav2 + nii-yamagishilab/mms-300m-anti-deepfake',
                         'revision': f'fusion {spec["version"]} · {sha[:12]}', 'fusion_sha256': sha,
                         'components': spec['models']}
        self.parameters = self.pella.parameters + self.adf.parameters
        self.preprocessing = {'normalization': 'each model’s own upstream normalization, per 4 s window',
                              'padding': 'pellav2 zero-pads short windows; AntiDeepfake scores them at their own length',
                              'window_tail': 'complete 4 s windows; a remaining tail gets a full window anchored at the file end',
                              'aggregation': 'arithmetic mean of fused window probabilities',
                              'window_samples': self.window_samples,
                              'fusion': (f'z = {self.beta[0]:.4f} + {self.beta[1]:.4f}·a + {self.beta[2]:.4f}·b'
                                         + (f' + {self.beta[3]:.4f}·n + {self.beta[4]:.4f}·a·n + {self.beta[5]:.4f}·b·n' if self.noise_aware else '')
                                         + f'; a = pellav2 logit, b = AntiDeepfake logit (clipped to ±{CLIP:g})'
                                         + ('; n = noise index from the window’s 20 ms frame-level spread' if self.noise_aware else ''))}

    def spans(self, x):
        from .forensics import anchored_spans
        return anchored_spans(len(x), self.window_samples)

    def score(self, x):
        pella = self.pella.score(x)
        native = self.adf.logits(x)
        adf_logit = float(native[0] - native[1])
        features = [pella['logits'][1], adf_logit] + ([noise_index(x)] if self.noise_aware else [])
        z = float((design(features) @ self.beta)[0])
        p = float(sigmoid(z))
        if not np.isfinite(z):
            raise ValueError('Fusion returned a nonfinite logit.')
        return {'synthetic_score': p, 'human_score': 1 - p, 'logits': [0.0, z],
                'components': {'pellav2': pella['synthetic_score'], 'antideepfake': float(sigmoid(adf_logit))},
                **({'noise_index': features[2]} if self.noise_aware else {})}
