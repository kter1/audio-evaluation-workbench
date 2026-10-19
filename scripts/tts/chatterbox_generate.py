"""Generate cloned speech with Chatterbox-Turbo. Runs in data/tts-envs/chatterbox, never imports lab.

Usage: python chatterbox_generate.py jobs.json
Each job: {"id", "ref_wav", "text", "seed", "output"}. Existing outputs are skipped.
Chatterbox embeds its imperceptible Perth watermark in every output; that is left unchanged.
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
    print(f'chatterbox: {len(pending)} of {len(jobs)} clips to generate', flush=True)
    if not pending:
        return
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    for i, job in enumerate(pending, 1):
        torch.manual_seed(job['seed'])
        wav = model.generate(job['text'], audio_prompt_path=job['ref_wav'])
        output = Path(job['output'])
        temporary = output.with_suffix('.part.wav')
        sf.write(temporary, wav.squeeze(0).numpy(), model.sr, subtype='PCM_16')
        os.replace(temporary, output)
        print(f'chatterbox {i}/{len(pending)} {job["id"]}', flush=True)
        del wav
        release()


if __name__ == '__main__':
    main()
