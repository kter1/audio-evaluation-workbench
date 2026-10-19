"""Build the frozen modern-voice evaluation set (protocol v3).

Stages are idempotent; run them in order:
  plan      choose LibriSpeech speakers, utterances, and cloning jobs (seeded; no detector involved)
  generate  run Chatterbox-Turbo and F5-TTS in their own Python 3.11 environments
  gate      Whisper transcription of every clip; WER > 0.2 is excluded before any detector scoring
  channels  MP3 64 kbps and 8 kHz mu-law telephone versions of the selected test clips
  freeze    write data/evaluation/modern-manifest.json (never overwritten) and docs/modern-set-audit.json

Matched design: each cloned fake reads the transcript of a held-back utterance by the same
speaker whose different utterance is the voice prompt; that held-back utterance is a real clip.
"""
import argparse
from collections import defaultdict
import hashlib
from math import ceil
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.evaluation import wer
from lab.store import digest

LIBRI = ROOT / 'data/librispeech/LibriSpeech'
WORK = ROOT / 'data/modern'
PLAN = WORK / 'plan.json'
GATE = WORK / 'gate.json'
MANIFEST = ROOT / 'data/evaluation/modern-manifest.json'
AUDIT = ROOT / 'docs/modern-set-audit.json'
SEED = 20261019
CALIBRATION_SPEAKERS = 12
EXTRA = 1.15
# test-other is noisier, so more real candidates fail ASR; raised after the gate, before any detector scoring.
EXTRA_BY_CELL = {'test/real-other': 1.5}
WER_LIMIT = .2
CHUNK = 15
COUNTS = {'calibration/real-clean': 60, 'calibration/chatterbox': 30, 'calibration/f5': 30,
          'test/real-clean': 60, 'test/real-other': 60, 'test/chatterbox': 60, 'test/f5': 60}
WHISPER = {'model': 'openai/whisper-base.en', 'revision': '911407f4214e0e1d82085af863093ec0b66f9cd6'}
GENERATORS = {
    'chatterbox': {'package': 'chatterbox-tts', 'model': 'ResembleAI/chatterbox-turbo (ChatterboxTurboTTS)',
                   'settings': {'temperature': .8, 'top_p': .95, 'top_k': 1000, 'repetition_penalty': 1.2,
                                'norm_loudness': True, 'watermark': 'Perth watermark applied by default; unchanged'}},
    'f5': {'package': 'f5-tts', 'model': 'SWivid/F5-TTS F5TTS_v1_Base',
           'settings': {'nfe_step': 32, 'cfg_strength': 2, 'sway_sampling_coef': -1, 'speed': 1.0}},
}
LIBRISPEECH = {'url': 'https://www.openslr.org/12', 'license': 'CC BY 4.0',
               'md5': {'test-clean.tar.gz': '32fa31d27d2e1cad72775fee3f4849a9',
                       'test-other.tar.gz': 'fb5a50374b501bb3bac4815ee91d3135'}}


def sentence(text):
    """LibriSpeech transcripts are upper case without punctuation; TTS reads sentence case."""
    text = re.sub(r"\bi\b", 'I', text.strip().lower())
    return text[:1].upper() + text[1:] + '.'


def utterances(subset, root=LIBRI):
    rows = []
    for transcript in sorted((root / subset).glob('*/*/*.trans.txt')):
        for line in transcript.read_text().splitlines():
            utt, text = line.split(' ', 1)
            path = transcript.parent / f'{utt}.flac'
            rows.append({'utt': utt, 'speaker': utt.split('-')[0], 'text': text,
                         'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                         'duration': sf.info(path).duration})
    return rows


def _round_robin(speakers, pools, cursor, n):
    chosen = []
    while len(chosen) < n:
        progressed = False
        for s in speakers:
            if len(chosen) == n:
                break
            if cursor[s] < len(pools[s]):
                chosen.append(pools[s][cursor[s]])
                cursor[s] += 1
                progressed = True
        if not progressed:
            raise ValueError('Not enough eligible utterances for the requested counts.')
    return chosen


def make_plan(clean, other, seed=SEED, calibration_speakers=CALIBRATION_SPEAKERS, counts=COUNTS, extra=EXTRA):
    """Seeded, speaker-disjoint plan. Returns candidates in selection order per cell."""
    rng = random.Random(seed)
    want = {cell: ceil(n * EXTRA_BY_CELL.get(cell, extra)) for cell, n in counts.items()}
    by_speaker = defaultdict(list)
    for u in sorted(clean, key=lambda u: u['utt']):
        by_speaker[u['speaker']].append(u)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    groups = {'calibration': speakers[:calibration_speakers], 'test': speakers[calibration_speakers:]}
    refs, pools = {}, {}
    for s in speakers:
        utts = list(by_speaker[s])
        rng.shuffle(utts)
        ref = next((u for u in utts if 6 <= u['duration'] <= 11.5), None)
        if ref is None:
            raise ValueError(f'Speaker {s} has no 6–11.5 s voice prompt.')
        refs[s] = ref
        pools[s] = [u for u in utts if u is not ref and 3 <= u['duration'] <= 15]
    cases, order = [], 0

    def fake(split, generator, s, u, k):
        ref = refs[s]
        return {'id': f'{split[:3]}-{generator}-{k:03d}', 'cell': f'{split}/{generator}', 'split': split,
                'label': 'synthetic', 'generator': generator, 'source': 'test-clean', 'speaker': s,
                'source_utt': u['utt'], 'text': sentence(u['text']), 'ref_utt': ref['utt'],
                'ref_text': sentence(ref['text']), 'ref_path': ref['path'], 'seed': seed + k,
                'path': f'data/modern/audio/{split[:3]}-{generator}-{k:03d}.wav'}

    def real(split, source, u, k):
        return {'id': f'{split[:3]}-real-{source[5:]}-{k:03d}', 'cell': f'{split}/real-{source[5:]}', 'split': split,
                'label': 'human', 'generator': 'bonafide', 'source': source, 'speaker': u['speaker'],
                'source_utt': u['utt'], 'text': sentence(u['text']), 'path': u['path']}

    for split in ('calibration', 'test'):
        cursor = {s: 0 for s in groups[split]}
        triplets = _round_robin(groups[split], pools, cursor, max(want[f'{split}/chatterbox'], want[f'{split}/f5']))
        for k, u in enumerate(triplets):
            for generator in GENERATORS:
                if k < want[f'{split}/{generator}']:
                    cases.append({**fake(split, generator, u['speaker'], u, k), 'order': order}); order += 1
        reals = triplets + _round_robin(groups[split], pools, cursor, max(0, want[f'{split}/real-clean'] - len(triplets)))
        for k, u in enumerate(reals[:want[f'{split}/real-clean']]):
            cases.append({**real(split, 'test-clean', u, k), 'order': order}); order += 1
    if 'test/real-other' in want:
        by_other = defaultdict(list)
        for u in sorted(other, key=lambda u: u['utt']):
            if 3 <= u['duration'] <= 15:
                by_other[u['speaker']].append(u)
        other_speakers = sorted(by_other)
        rng.shuffle(other_speakers)
        for s in other_speakers:
            rng.shuffle(by_other[s])
        picked = _round_robin(other_speakers, by_other, {s: 0 for s in other_speakers}, want['test/real-other'])
        for k, u in enumerate(picked):
            cases.append({**real('test', 'test-other', u, k), 'order': order}); order += 1
    return {'seed': seed, 'counts': counts, 'extra': extra, 'speakers': groups,
            'refs': {s: refs[s]['utt'] for s in speakers}, 'cases': cases}


def select(plan, gate, counts=None):
    """First N clips per cell, in seeded plan order, that exist and pass the ASR gate."""
    counts = counts or COUNTS
    selected, excluded, unused = [], [], []
    taken = defaultdict(int)
    for case in sorted(plan['cases'], key=lambda c: c['order']):
        result = gate.get(case['id'])
        if taken[case['cell']] >= counts[case['cell']]:
            unused.append(case['id'])
        elif result is None:
            excluded.append({'id': case['id'], 'cell': case['cell'], 'reason': 'not generated or not gated'})
        elif result.get('error'):
            excluded.append({'id': case['id'], 'cell': case['cell'], 'reason': result['error']})
        elif result['wer'] > WER_LIMIT:
            excluded.append({'id': case['id'], 'cell': case['cell'], 'reason': f'ASR WER {result["wer"]:.3f} > {WER_LIMIT}',
                             'wer': result['wer'], 'transcript': result['hypothesis']})
        else:
            selected.append(case)
            taken[case['cell']] += 1
    short = {cell: n - taken[cell] for cell, n in counts.items() if taken[cell] < n}
    return selected, excluded, unused, short


def stage_plan():
    if PLAN.exists():
        print('Plan already exists:', PLAN.relative_to(ROOT))
        return
    plan = make_plan(utterances('test-clean'), utterances('test-other'))
    WORK.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Planned {len(plan["cases"])} candidate clips:', dict(sorted(
        {cell: sum(c['cell'] == cell for c in plan['cases']) for cell in COUNTS}.items())))


def stage_generate(only=None):
    plan = json.loads(PLAN.read_text())
    (WORK / 'refs').mkdir(parents=True, exist_ok=True)
    (WORK / 'audio').mkdir(parents=True, exist_ok=True)
    for case in plan['cases']:
        if case['label'] == 'synthetic':
            ref = WORK / 'refs' / f'{case["ref_utt"]}.wav'
            if not ref.exists():  # F5 reads prompts through pydub, which needs WAV without system ffmpeg
                audio, rate = sf.read(ROOT / case['ref_path'], dtype='int16')
                sf.write(ref, audio, rate, subtype='PCM_16')
    env = {**os.environ, 'HF_HOME': str(ROOT / 'data/hf-cache'), 'TTS_HOME': str(ROOT / 'data/hf-cache/tts'),
           'PYTORCH_ENABLE_MPS_FALLBACK': '1'}
    for generator in GENERATORS:
        if only and generator != only:
            continue
        jobs = [{'id': c['id'], 'ref_wav': str(WORK / 'refs' / f'{c["ref_utt"]}.wav'), 'ref_text': c['ref_text'],
                 'text': c['text'], 'seed': c['seed'], 'output': str(ROOT / c['path'])}
                for c in plan['cases'] if c['generator'] == generator]
        jobs = [j for j in jobs if not Path(j['output']).exists()]
        python = ROOT / f'data/tts-envs/{generator}/bin/python'
        # Fresh process per chunk: bounds any per-clip memory growth in the generators.
        for start in range(0, len(jobs), CHUNK):
            jobs_path = WORK / f'jobs-{generator}.json'
            jobs_path.write_text(json.dumps(jobs[start:start + CHUNK], indent=2) + '\n')
            subprocess.run([str(python), str(ROOT / f'scripts/tts/{generator}_generate.py'), str(jobs_path)],
                           cwd=ROOT, env=env, check=True)


def stage_gate():
    os.environ.setdefault('HF_HOME', str(ROOT / 'data/hf-cache'))
    from transformers import pipeline
    from lab.forensics import RATE, mono16, read_audio
    plan = json.loads(PLAN.read_text())
    gate = json.loads(GATE.read_text()) if GATE.exists() else {}
    asr = pipeline('automatic-speech-recognition', model=WHISPER['model'], revision=WHISPER['revision'], device='cpu')
    todo = [c for c in plan['cases'] if c['id'] not in gate]
    for i, case in enumerate(todo, 1):
        path = ROOT / case['path']
        if not path.exists():
            continue  # not generated yet; gate again after generation
        try:
            audio, rate, _ = read_audio(path.read_bytes())
            hypothesis = asr({'raw': mono16(audio, rate), 'sampling_rate': RATE})['text']
            gate[case['id']] = {'wer': wer(case['text'], hypothesis), 'hypothesis': hypothesis.strip()}
        except ValueError as error:
            gate[case['id']] = {'error': f'Unreadable audio: {error}'}
        if i % 25 == 0 or i == len(todo):
            GATE.write_text(json.dumps(gate, indent=2, sort_keys=True) + '\n')
            print(f'gate {i}/{len(todo)}', flush=True)
    GATE.write_text(json.dumps(gate, indent=2, sort_keys=True) + '\n')
    selected, excluded, _, short = select(plan, gate)
    print(f'Selected {len(selected)}; excluded {len(excluded)}; short cells: {short or "none"}')


CHANNELS = ('mp3', 'phone')
FFMPEG_CHANNELS = {'mp3': ['-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k'],
                   'phone': ['-ac', '1', '-ar', '8000', '-c:a', 'pcm_mulaw']}
NOISE_CHANNELS = {'noise20': 20}  # white Gaussian noise at this SNR (dB) relative to the clip's RMS


def channel_paths(case, work=None):
    stem = (work or WORK) / 'channels' / case['id']
    return {name: stem.with_suffix('.mp3') if name == 'mp3' else Path(f'{stem}-{name}.wav') for name in CHANNELS}


def render(source, target, name, key):
    """Write one channel version of `source`; `key` seeds the noise so renders are reproducible."""
    from lab.forensics import read_audio
    temporary = target.with_name('part-' + target.name)
    if name in NOISE_CHANNELS:
        import numpy as np
        audio, rate = sf.read(source, dtype='float64', always_2d=True)
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        noise = rng.normal(0, 1, audio.shape)
        noise *= np.sqrt(np.mean(audio**2)) / np.sqrt(np.mean(noise**2)) * 10**(-NOISE_CHANNELS[name]/20)
        sf.write(temporary, np.clip(audio + noise, -1, 1), rate, subtype='PCM_16', format='WAV')
    else:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                        *FFMPEG_CHANNELS[name], str(temporary)], check=True)
    read_audio(temporary.read_bytes())  # must decode through the lab's own reader
    temporary.replace(target)


def stage_channels():
    plan = json.loads(PLAN.read_text())
    selected, _, _, short = select(plan, json.loads(GATE.read_text()))
    if short:
        raise ValueError(f'Cells are short of passing clips: {short}. Generate or gate more before channels.')
    (WORK / 'channels').mkdir(parents=True, exist_ok=True)
    for case in selected:
        if case['split'] != 'test':
            continue
        for name, target in channel_paths(case).items():
            if not target.exists():
                render(ROOT / case['path'], target, name, case['id'])
    print('Channel variants ready.')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_version(generator):
    package = GENERATORS[generator]['package']
    for line in (ROOT / f'scripts/tts/{generator}.lock').read_text().splitlines():
        if line.lower().startswith(package + '=='):
            return line.split('==', 1)[1]
    return None


def manifest_body(plan, selected, excluded, unused):
    from lab.forensics import read_audio
    cases = []
    for case in sorted(selected, key=lambda c: c['order']):
        blob = (ROOT / case['path']).read_bytes()
        _, _, info = read_audio(blob)
        entry = {k: case[k] for k in ('id', 'cell', 'split', 'label', 'generator', 'source', 'speaker',
                                      'source_utt', 'text', 'path') if k in case}
        if case['label'] == 'synthetic':
            entry.update({'ref_utt': case['ref_utt'], 'seed': case['seed']})
        entry.update({'sha256': info['sha256'], 'duration': info['duration_seconds'], 'sample_rate': info['original_rate']})
        if case['split'] == 'test':
            entry['channels'] = {name: {'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)}
                                 for name, path in channel_paths(case).items()}
        cases.append(entry)
    return {'version': '3.0', 'seed': plan['seed'], 'counts': plan['counts'],
            'rules': {'speakers': f'LibriSpeech test-clean speakers shuffled; first {CALIBRATION_SPEAKERS} calibration, rest test',
                      'cloning': 'voice prompt = one 6–11.5 s utterance per speaker; text = a different 3–15 s utterance by the same speaker, whose real recording is a human clip',
                      'selection': f'first N per cell in seeded order with readable audio and Whisper WER <= {WER_LIMIT}; {EXTRA:.2f}x candidates ({EXTRA_BY_CELL} for noisier cells, extended in the same seeded order after the gate); detector scores never used',
                      'channels': 'test split only: MP3 64 kbps mono; 8 kHz mono G.711 mu-law WAV; via the bundled imageio-ffmpeg'},
            'generators': {g: {**spec, 'version': package_version(g)} for g, spec in GENERATORS.items()},
            'asr_gate': {**WHISPER, 'wer_limit': WER_LIMIT, 'normalization': 'lowercase words [a-z0-9\']'},
            'librispeech': LIBRISPEECH, 'speakers': plan['speakers'],
            'cases': cases, 'excluded': excluded, 'unused_candidates': len(unused)}


def stage_freeze():
    if MANIFEST.exists():
        print('Frozen manifest already exists:', MANIFEST.relative_to(ROOT))
        return
    plan = json.loads(PLAN.read_text())
    selected, excluded, unused, short = select(plan, json.loads(GATE.read_text()))
    if short:
        raise ValueError(f'Cells are short of passing clips: {short}')
    body = manifest_body(plan, selected, excluded, unused)
    body['sha256'] = digest(body)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(body, indent=2) + '\n')

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items() if k != 'path'}
        return [portable(v) for v in value] if isinstance(value, list) else value
    AUDIT.write_text(json.dumps({'manifest_sha256': body['sha256'], 'note': 'Portable record of the frozen modern-voice set; local audio paths removed.',
                                 **portable({k: v for k, v in body.items() if k != 'sha256'})}, indent=2) + '\n')
    print(f'Frozen {len(body["cases"])} clips; excluded {len(excluded)}; SHA-256 {body["sha256"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['plan', 'generate', 'gate', 'channels', 'freeze'])
    parser.add_argument('--generator', choices=list(GENERATORS), help='generate only this generator')
    args = parser.parse_args()
    {'plan': stage_plan, 'gate': stage_gate, 'channels': stage_channels, 'freeze': stage_freeze,
     'generate': lambda: stage_generate(args.generator)}[args.stage]()


if __name__ == '__main__':
    main()
