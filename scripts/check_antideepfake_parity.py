"""Check the transformers port of AntiDeepfake against the official fairseq code on whole clips.

Runs scripts/reference/antideepfake_fairseq.py in data/tts-envs/fairseq (Python 3.9, fairseq at the
authors' pinned commit), then scores the same clips with lab.antideepfake. Both sides receive the
identical 16 kHz mono signal (the lab's own resampling, written as float WAV), so this isolates the
model; the official script's torchaudio resampler differs from the lab's SciPy one for non-16 kHz input.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.antideepfake import MODEL_DIR, AntiDeepfakeDetector
from lab.forensics import mono16, read_audio

LOGIT_TOLERANCE = 1e-3
PROB_TOLERANCE = 1e-4


def default_clips():
    clean = sorted((ROOT / 'data/librispeech/LibriSpeech/test-clean').glob('*/*/*.flac'))[:5]
    fakes = sorted((ROOT / 'data/modern/audio').glob('tes-chatterbox-00*.wav'))[:2]
    fakes += sorted((ROOT / 'data/modern/audio').glob('tes-f5-00*.wav'))[:1]
    return [p for p in clean + fakes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clips', nargs='*', type=Path)
    clips = parser.parse_args().clips or default_clips()
    ours = AntiDeepfakeDetector()  # verifies pinned checksums before anything runs
    signals = {}
    with tempfile.TemporaryDirectory() as tmp:
        inputs = []
        for clip in clips:
            audio, rate, _ = read_audio(clip.read_bytes())
            signals[clip] = mono16(audio, rate)
            inputs.append(Path(tmp) / f'{clip.stem}.wav')
            sf.write(inputs[-1], signals[clip], 16000, subtype='FLOAT')
        out = Path(tmp) / 'reference.json'
        subprocess.run([str(ROOT / 'data/tts-envs/fairseq/bin/python'), str(ROOT / 'scripts/reference/antideepfake_fairseq.py'),
                        str(MODEL_DIR), str(out), *map(str, inputs)], check=True, cwd=ROOT)
        reference = {clip: json.loads(out.read_text())[str(path)] for clip, path in zip(clips, inputs)}
    worst_logit = worst_prob = 0.0
    for clip in clips:
        x = signals[clip]
        native = ours.logits(x, samples=len(x))
        mine = [float(v) for v in native]
        prob = float(ours.torch.softmax(native, dim=-1)[0])
        ref = reference[clip]
        dl = max(abs(a - b) for a, b in zip(mine, ref['logits_fake_real']))
        dp = abs(prob - ref['fake_prob'])
        worst_logit, worst_prob = max(worst_logit, dl), max(worst_prob, dp)
        print(f'{clip.name}: fairseq fake {ref["fake_prob"]:.6f} · lab {prob:.6f} · |Δlogit| {dl:.2e}', flush=True)
    ok = worst_logit <= LOGIT_TOLERANCE and worst_prob <= PROB_TOLERANCE
    print(f'Largest |Δlogit| {worst_logit:.2e} (≤ {LOGIT_TOLERANCE:.0e}), |Δp| {worst_prob:.2e} (≤ {PROB_TOLERANCE:.0e}):', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
