"""Check that the lab's pellav2 adapter reproduces the upstream pellav2_infer.py scores.

Upstream preprocessing and model code run unmodified, except that the backbone is built
from its pinned configuration instead of downloading pretrained weights the checkpoint
replaces anyway. Both sides score the same centre 4 s crop (upstream's clip rule).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.forensics import mono16, read_audio
from lab.pella import MODEL_DIR, PellaV2Detector

TOLERANCE = 1e-5


def default_clips(limit=5):
    clips = []
    for path in sorted(Path('data/asvspoof/audio').glob('*.flac')):
        try:
            sf.read(path, frames=1)
            sf.read(path)
        except RuntimeError:
            continue  # upstream reads with SoundFile only
        clips.append(path)
        if len(clips) == limit:
            break
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clips', nargs='*', type=Path, help='16 kHz speech files readable by SoundFile')
    args = parser.parse_args()
    clips = args.clips or default_clips()
    if not clips:
        raise SystemExit('No readable clips found; pass 16 kHz speech files.')
    ours = PellaV2Detector()  # verifies every pinned checksum, including pellav2_infer.py
    spec = importlib.util.spec_from_file_location('pellav2_infer_upstream', MODEL_DIR / 'pellav2_infer.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    config = json.loads((MODEL_DIR / 'xls-r-300m-config.json').read_text())
    with mock.patch.object(Wav2Vec2Model, 'from_pretrained', lambda *a, **k: Wav2Vec2Model(Wav2Vec2Config.from_dict(config))):
        reference = upstream.Detector()
    reference.load_state_dict(torch.load(MODEL_DIR / 'pellav2_detector.pt', map_location='cpu', weights_only=True))
    reference.eval()
    worst = 0.0
    for path in clips:
        with torch.no_grad():
            p_upstream = torch.sigmoid(reference(torch.from_numpy(upstream.load_clip(str(path)))[None])).item()
        audio, rate, _ = read_audio(path.read_bytes())
        x = mono16(audio, rate)
        if len(x) >= upstream.CROP:
            offset = (len(x) - upstream.CROP) // 2
            x = x[offset:offset + upstream.CROP]
        p_ours = ours.score(x)['synthetic_score']
        difference = abs(p_ours - p_upstream)
        worst = max(worst, difference)
        print(f'{path.name}: upstream {p_upstream:.6f} · lab {p_ours:.6f} · |Δ| {difference:.2e}', flush=True)
    print(f'Largest difference {worst:.2e} (tolerance {TOLERANCE:.0e}):', 'PASS' if worst <= TOLERANCE else 'FAIL')
    return 0 if worst <= TOLERANCE else 1


if __name__ == '__main__':
    raise SystemExit(main())
