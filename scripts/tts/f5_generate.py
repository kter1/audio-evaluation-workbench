"""Generate cloned speech with F5-TTS v1 Base. Runs in data/tts-envs/f5, never imports lab.

Usage: python f5_generate.py jobs.json
Each job: {"id", "ref_wav", "ref_text", "text", "seed", "output"}. Existing outputs are skipped.
"""
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch


def release():
    """MPS keeps per-clip buffers; without this the process grew to 36 GB over ~80 clips."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    pending = [j for j in jobs if not Path(j['output']).exists()]
    print(f'f5: {len(pending)} of {len(jobs)} clips to generate', flush=True)
    if not pending:
        return
    import torchaudio
    from f5_tts.api import F5TTS

    def load(path, *args, **kwargs):
        # torchaudio >= 2.9 loads through torchcodec, which needs system FFmpeg; prompts are plain WAV.
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        return torch.from_numpy(audio.T.copy()), rate
    torchaudio.load = load
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = F5TTS(model='F5TTS_v1_Base', device=device)
    for i, job in enumerate(pending, 1):
        wav, rate, _ = model.infer(ref_file=job['ref_wav'], ref_text=job['ref_text'], gen_text=job['text'],
                                   seed=job['seed'], show_info=lambda *a: None, progress=None)
        output = Path(job['output'])
        temporary = output.with_suffix('.part.wav')
        sf.write(temporary, wav, rate, subtype='PCM_16')
        os.replace(temporary, output)
        print(f'f5 {i}/{len(pending)} {job["id"]}', flush=True)
        del wav
        release()


if __name__ == '__main__':
    main()
