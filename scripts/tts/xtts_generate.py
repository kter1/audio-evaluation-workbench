"""Generate cloned speech with Coqui XTTS-v2. Runs in data/tts-envs/xtts, never imports lab.

Usage: python xtts_generate.py jobs.json
Each job: {"id", "ref_wav", "text", "seed", "output"}. Existing outputs are skipped.
XTTS-v2 is under the Coqui Public Model License (non-commercial). The project owner accepted it on
2026-10-19 for local research clips that stay under data/ and are never redistributed.
"""
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

os.environ['COQUI_TOS_AGREED'] = '1'


def release():
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    pending = [j for j in jobs if not Path(j['output']).exists()]
    print(f'xtts: {len(pending)} of {len(jobs)} clips to generate', flush=True)
    if not pending:
        return
    from TTS.api import TTS
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
    # Unsupported MPS ops fall back to CPU via PYTORCH_ENABLE_MPS_FALLBACK (set by the driver).
    device = os.environ.get('XTTS_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu')
    tts = tts.to(device)
    rate = tts.synthesizer.output_sample_rate
    for i, job in enumerate(pending, 1):
        torch.manual_seed(job['seed'])
        wav = tts.tts(text=job['text'], speaker_wav=job['ref_wav'], language='en')
        output = Path(job['output'])
        temporary = output.with_suffix('.part.wav')
        sf.write(temporary, wav, rate, subtype='PCM_16')
        os.replace(temporary, output)
        print(f'xtts {i}/{len(pending)} {job["id"]}', flush=True)
        del wav
        release()


if __name__ == '__main__':
    main()
