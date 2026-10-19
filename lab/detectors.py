"""One registry for the selectable detectors, their setup commands, and install checks."""
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# kind -> (module, class, file that exists once installed, setup command)
DETECTORS = {
    'fusion': ('lab.fusion', 'FusionDetector', 'lab/fusion-v4.json', 'scripts/setup_pellav2.py && .venv/bin/python scripts/setup_antideepfake.py'),
    'antideepfake': ('lab.antideepfake', 'AntiDeepfakeDetector', 'data/models/antideepfake/model.safetensors', 'scripts/setup_antideepfake.py'),
    'pellav2': ('lab.pella', 'PellaV2Detector', 'data/models/pellav2/pellav2_detector.pt', 'scripts/setup_pellav2.py'),
    'aasist': ('lab.aasist', 'AASISTDetector', 'data/models/aasist/model.pth', 'scripts/setup_aasist.py'),
    'wav2vec2': ('lab.forensics', 'Detector', 'data/models/wav2vec2-detector/manifest.json', 'scripts/setup_detector.py'),
}
LABELS = {'fusion': 'pellav2 + AntiDeepfake fusion', 'antideepfake': 'AntiDeepfake MMS-300M', 'pellav2': 'pellav2',
          'aasist': 'AASIST (2019)', 'wav2vec2': 'Wav2Vec2-XLSR'}


def installed(kind):
    if kind == 'fusion':  # needs both component checkpoints and the frozen fusion weights
        return all((ROOT / DETECTORS[k][2]).exists() for k in ('fusion', 'pellav2', 'antideepfake'))
    return (ROOT / DETECTORS[kind][2]).exists()


def setup_command(kind):
    return f'.venv/bin/python {DETECTORS[kind][3]}'


def load(kind):
    if kind not in DETECTORS:
        raise ValueError(f'Unknown detector: {kind}')
    module, name = DETECTORS[kind][:2]
    return getattr(import_module(module), name)()
