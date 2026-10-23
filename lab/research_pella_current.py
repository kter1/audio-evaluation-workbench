"""Isolated current Pellav2 candidate; never selected by the product.

The pinned author's feature/forward methods and official TorchAudio filterbank
definitions are executed only after file authentication. The backbone is built
from its pinned configuration, with every weight supplied by the checkpoint.
No Hub access, package installation, or unrestricted pickle loading occurs.
"""
import ast
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'Sadanie/pellav2-audio-deepfake-detector'
REVISION = '808e16a49526178f8e1e6276725dfd14f218ec82'
BACKBONE_REVISION = 'da985ba0987f70aaeb84a80f2851cfac8c697a7b'
KALDI_REVISION = 'a224ab24a7f4797f6707051257265e223e12576f'
SR, WINDOW, TARGET_LUFS = 16000, 64000, -23.0
FILES = {
    'pellav2_detector.pt': (2322248702, '2279c24f6e4b672c85eb2593f93cbbad83a7c1e9486fe3bcdc246c13fcca80e6'),
    'pellav2_infer.py': (4496, 'bf13cc97e5f0466b6dd6797afce7abe9befc61ec8817c05cbe505f88a29c6a67'),
    'w2v-bert-config.json': (1874, 'f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675'),
    'README.md': (484, '88351224bc14400fa61c8f22c264d50b3430661a07f87136be201a01a8ab14e2'),
    'kaldi.py': (36666, '5cbea1a584ddea748f6f68a621d794e13334e88d9faa1d40986f7af32f196d29'),
    'TORCHAUDIO-LICENSE': (1338, '93a58861a858cc108e6b6b833e08e76e8b2a66339e4a8007c8a5a8c1ff9c40d6'),
}
KALDI_FUNCTIONS = (
    '_get_epsilon', '_next_power_of_2', '_get_strided',
    '_feature_window_function', '_get_log_energy',
    '_get_waveform_and_window_properties', '_get_window',
    '_subtract_column_mean', 'inverse_mel_scale_scalar', 'inverse_mel_scale',
    'mel_scale_scalar', 'mel_scale', 'vtln_warp_freq', 'vtln_warp_mel_freq',
    'get_mel_banks', 'fbank',
)


def verify_file(directory, name):
    """Authenticate source and large checkpoints with bounded streaming reads."""
    path = Path(directory)/name
    size, expected = FILES[name]
    if path.stat().st_size != size:
        raise ValueError('Current Pellav2 file size mismatch: '+name)
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Current Pellav2 SHA-256 mismatch: '+name)
    return {'bytes': size, 'sha256': expected}


def loudness(x):
    """Pinned author's integrated-LUFS path, including its RMS fallback."""
    try:
        import pyloudnorm as pyln
        if len(x) >= int(.4*SR):
            value = pyln.Meter(SR).integrated_loudness(x.astype(np.float64))
            if np.isfinite(value):
                return float(value)
    except Exception:
        pass
    return float(20*np.log10(np.sqrt(np.mean(x**2))+1e-9)-.691)


def preprocess(samples):
    """Author's load_clip from supplied finite mono16k audio, without decoding.

    Keep DC removal and highpass before cropping; normalize loudness on the crop
    before right padding. Features then ignore the resulting exact-zero tail.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite mono16k audio')
    from scipy.signal import butter, sosfiltfilt
    x = x-x.mean()
    if len(x) > 3*SR//100:
        x = sosfiltfilt(butter(2, 20., btype='highpass', fs=SR, output='sos'), x)
    x = x.astype(np.float32)
    if len(x) > WINDOW:
        offset = (len(x)-WINDOW)//2
        x = x[offset:offset+WINDOW]
    gain = float(np.clip(TARGET_LUFS-loudness(x), -40., 40.))
    x = np.clip(x*10**(gain/20), -1., 1.).astype(np.float32)
    if len(x) < WINDOW:
        x = np.concatenate([x, np.zeros(WINDOW-len(x), np.float32)])
    return x


def load_kaldi(directory):
    """Load only authenticated pure-Torch filterbank definitions, no imports.

    The unused MFCC/DCT path requires native TorchAudio; fbank does not. The
    original BSD license is kept alongside the complete unmodified source.
    """
    verify_file(directory, 'kaldi.py')
    verify_file(directory, 'TORCHAUDIO-LICENSE')
    import torch
    path = Path(directory)/'kaldi.py'
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name in KALDI_FUNCTIONS]
    if {n.name for n in definitions} != set(KALDI_FUNCTIONS):
        raise ValueError('Pinned Kaldi definitions are incomplete')
    namespace = {
        'math': math, 'torch': torch, 'Tensor': torch.Tensor, 'Tuple': Tuple,
        'EPSILON': torch.tensor(torch.finfo(torch.float).eps),
        'MILLISECONDS_TO_SECONDS': .001, 'HAMMING': 'hamming',
        'HANNING': 'hanning', 'POVEY': 'povey',
        'RECTANGULAR': 'rectangular', 'BLACKMAN': 'blackman',
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in KALDI_FUNCTIONS})


def author_detector_type(directory, kaldi=None):
    """Exact feature/forward code; replace only network-loading construction.

    The one import inside features is removed and bound to the verified official
    filterbank namespace. No global sys.modules stubs or top-level source code
    are executed. Remaining methods are the pinned author's unchanged AST.
    """
    verify_file(directory, 'pellav2_infer.py')
    import torch
    path = Path(directory)/'pellav2_infer.py'
    tree = ast.parse(path.read_text())
    detector = next(n for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == 'Detector')
    detector.body = [n for n in detector.body
                     if isinstance(n, ast.FunctionDef) and n.name in ('features', 'forward')]
    if {n.name for n in detector.body} != {'features', 'forward'}:
        raise ValueError('Pinned Pellav2 definitions are incomplete')
    features = next(n for n in detector.body if n.name == 'features')
    imports = [n for n in features.body if isinstance(n, ast.Import)]
    if len(imports) != 1 or imports[0].names[0].name != 'torchaudio.compliance.kaldi':
        raise ValueError('Pinned Pellav2 feature import differs')
    features.body.remove(imports[0])
    namespace = {'torch': torch, 'nn': torch.nn, 'SR': SR,
                 'K': kaldi if kaldi is not None else load_kaldi(directory)}
    exec(compile(ast.Module(body=[detector], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Detector']


def build_model(directory, config):
    """Construct all parameters locally; the checkpoint must supply all weights."""
    import torch
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
    AuthorDetector = author_detector_type(directory)

    class OfflineDetector(AuthorDetector):
        def __init__(self):
            super().__init__()
            self.backbone = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_dict(config))
            self.layer_weights = torch.nn.Parameter(
                torch.zeros(self.backbone.config.num_hidden_layers+1))
            self.head = torch.nn.Linear(self.backbone.config.hidden_size, 1)

    return OfflineDetector()


def validate_state(state):
    """Fail before model construction when authenticated saved weights are bad."""
    import torch
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError('Current Pellav2 checkpoint contains a non-tensor: '+name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Current Pellav2 checkpoint contains nonfinite weights: '+name)


class PellaCurrentDetector:
    threshold = .5
    preprocessing = {
        'sample_rate': SR, 'samples': WINDOW,
        'normalization': 'DC remove, 20Hz 2nd-order zero-phase highpass, center crop, -23LUFS with gain clipped +/-40dB, clip +/-1',
        'padding': 'right zero pad; feature extraction ends at last nonzero sample',
        'features': 'official TorchAudio Kaldi fbank: 80 mel bins, Povey, dither 0, unbiased sample variance normalization, adjacent pairs to 160 features',
        'aggregation': 'learned softmax layer weights; masked temporal mean; linear head',
        'score_direction': 'sigmoid native logit; higher indicates synthetic speech',
    }

    def __init__(self, directory=None):
        directory = Path(directory) if directory else ROOT/'data/research-tools/pella-current'/REVISION
        # Authenticate everything before importing dependencies, executing source,
        # or restricted checkpoint deserialization.
        files = {name: verify_file(directory, name) for name in FILES}
        import torch
        self.torch = torch
        config = json.loads((directory/'w2v-bert-config.json').read_text())
        state = torch.load(directory/'pellav2_detector.pt', map_location='cpu',
                           weights_only=True, mmap=True)
        validate_state(state)
        self.model = build_model(directory, config)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval()
        try:
            from importlib.metadata import version
            pyloudnorm = version('pyloudnorm')
        except Exception:
            pyloudnorm = None
        self.manifest = {
            'repository': REPOSITORY, 'revision': REVISION,
            'backbone': 'facebook/w2v-bert-2.0', 'backbone_revision': BACKBONE_REVISION,
            'kaldi_revision': KALDI_REVISION, 'license': 'MIT (publisher card)',
            'files': files, 'native_threshold': self.threshold,
            'pyloudnorm_version': pyloudnorm, 'preprocessing': self.preprocessing,
        }
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.timings = []

    def score(self, samples):
        started = time.perf_counter()
        waveform = self.torch.from_numpy(preprocess(samples))[None]
        with self.torch.inference_mode():
            features, mask = self.model.features(waveform)
            output = self.model(waveform)[0]
            score = float(self.torch.sigmoid(output))
        logit = float(output)
        if not np.isfinite(logit) or not 0 <= score <= 1:
            raise ValueError('Current Pellav2 returned an invalid score')
        self.timings.append({'seconds': time.perf_counter()-started,
                             'input_seconds': len(samples)/SR, 'model_seconds': 4.})
        return {
            'synthetic_score': score, 'human_score': 1-score, 'logits': [0., logit],
            'model_input_sha256': sha256(waveform.numpy().astype('<f4').tobytes()).hexdigest(),
            'feature_sha256': sha256(features.numpy().astype('<f4').tobytes()).hexdigest(),
            'feature_frames': int(mask.sum()),
        }
