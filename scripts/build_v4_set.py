"""Build the fresh protocol-v4 test set: new speakers, a held-out generator, four channels.

Reuses the v3 pipeline in build_modern_set.py with v4 settings. Stages, in order:
  plan | generate | gate | channels | freeze

Real speech: LibriSpeech dev-clean and dev-other, whose speakers appear in no earlier set.
Fakes clone dev-clean speakers reading a different utterance by the same speaker:
Chatterbox-Turbo and F5-TTS (generators seen in development) and XTTS-v2, which is held out
from all development and calibration. Channels: clean, MP3 64 kbps, 8 kHz mu-law telephone,
and white noise at 20 dB SNR.
"""
import argparse
from collections import defaultdict
from math import ceil
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.store import digest
from scripts import build_modern_set as m

SEED = 20261019
HELDOUT = 'xtts'
COUNTS = {'test/real-clean': 100, 'test/real-other': 100, 'test/chatterbox': 30, 'test/f5': 30, 'test/xtts': 60}
EXTRA = {'test/real-clean': 1.15, 'test/real-other': 1.5, 'test/chatterbox': 1.15, 'test/f5': 1.15, 'test/xtts': 1.3}
GENERATORS = {**m.GENERATORS,
              'xtts': {'package': 'coqui-tts', 'model': 'coqui/XTTS-v2 (tts_models/multilingual/multi-dataset/xtts_v2)',
                       'settings': {'language': 'en', 'device': 'mps', 'license': 'Coqui Public Model License (non-commercial), accepted by the project owner for local research clips'}}}
LIBRISPEECH = {'url': 'https://www.openslr.org/12', 'license': 'CC BY 4.0',
               'md5': {'dev-clean.tar.gz': '42e2234ba48799c1f50f24a7926300a1',
                       'dev-other.tar.gz': 'c8d0bcc9cca99d4f8b62fcc847357931'}}


def configure():
    """Point the shared v3 pipeline at v4 paths, cells, generators, and channels."""
    m.WORK = ROOT / 'data/modern-v4'
    m.PLAN = m.WORK / 'plan.json'
    m.GATE = m.WORK / 'gate.json'
    m.MANIFEST = ROOT / 'data/evaluation/v4-manifest.json'
    m.AUDIT = ROOT / 'docs/v4-set-audit.json'
    m.COUNTS = COUNTS
    m.GENERATORS = GENERATORS
    m.CHANNELS = ('mp3', 'phone', 'noise20')


def make_plan(clean, other, seed=SEED, counts=COUNTS, extra=EXTRA):
    rng = random.Random(seed)
    want = {cell: ceil(n * extra[cell]) for cell, n in counts.items()}
    by_speaker = defaultdict(list)
    for u in sorted(clean, key=lambda u: u['utt']):
        by_speaker[u['speaker']].append(u)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
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
    generators = [g for g in GENERATORS if f'test/{g}' in counts]
    cursor = {s: 0 for s in speakers}
    targets = m._round_robin(speakers, pools, cursor, max(want[f'test/{g}'] for g in generators))
    for k, u in enumerate(targets):
        for g in generators:
            if k < want[f'test/{g}']:
                ref = refs[u['speaker']]
                cases.append({'id': f'v4-{g}-{k:03d}', 'cell': f'test/{g}', 'split': 'test', 'label': 'synthetic',
                              'generator': g, 'source': 'dev-clean', 'speaker': u['speaker'], 'source_utt': u['utt'],
                              'text': m.sentence(u['text']), 'ref_utt': ref['utt'], 'ref_text': m.sentence(ref['text']),
                              'ref_path': ref['path'], 'seed': seed + k, 'path': f'data/modern-v4/audio/v4-{g}-{k:03d}.wav',
                              'order': order})
                order += 1
    reals = targets + m._round_robin(speakers, pools, cursor, max(0, want['test/real-clean'] - len(targets)))
    for k, u in enumerate(reals[:want['test/real-clean']]):
        cases.append({'id': f'v4-real-clean-{k:03d}', 'cell': 'test/real-clean', 'split': 'test', 'label': 'human',
                      'generator': 'bonafide', 'source': 'dev-clean', 'speaker': u['speaker'], 'source_utt': u['utt'],
                      'text': m.sentence(u['text']), 'path': u['path'], 'order': order})
        order += 1
    by_other = defaultdict(list)
    for u in sorted(other, key=lambda u: u['utt']):
        if 3 <= u['duration'] <= 15:
            by_other[u['speaker']].append(u)
    other_speakers = sorted(by_other)
    rng.shuffle(other_speakers)
    for s in other_speakers:
        rng.shuffle(by_other[s])
    for k, u in enumerate(m._round_robin(other_speakers, by_other, {s: 0 for s in other_speakers}, want['test/real-other'])):
        cases.append({'id': f'v4-real-other-{k:03d}', 'cell': 'test/real-other', 'split': 'test', 'label': 'human',
                      'generator': 'bonafide', 'source': 'dev-other', 'speaker': u['speaker'], 'source_utt': u['utt'],
                      'text': m.sentence(u['text']), 'path': u['path'], 'order': order})
        order += 1
    return {'seed': seed, 'counts': counts, 'extra': extra, 'heldout_generator': HELDOUT,
            'speakers': {'dev-clean': speakers, 'dev-other': other_speakers},
            'refs': {s: refs[s]['utt'] for s in speakers}, 'cases': cases}


def stage_plan():
    if m.PLAN.exists():
        print('Plan already exists:', m.PLAN.relative_to(ROOT))
        return
    plan = make_plan(m.utterances('dev-clean'), m.utterances('dev-other'))
    m.WORK.mkdir(parents=True, exist_ok=True)
    m.PLAN.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Planned {len(plan["cases"])} candidates:', {c: sum(x['cell'] == c for x in plan['cases']) for c in COUNTS})


def stage_freeze():
    if m.MANIFEST.exists():
        print('Frozen manifest already exists:', m.MANIFEST.relative_to(ROOT))
        return
    plan = json.loads(m.PLAN.read_text())
    selected, excluded, unused, short = m.select(plan, json.loads(m.GATE.read_text()))
    if short:
        raise ValueError(f'Cells are short of passing clips: {short}')
    body = m.manifest_body(plan, selected, excluded, unused)
    body.update({'version': '4.0', 'heldout_generator': HELDOUT, 'librispeech': LIBRISPEECH, 'speakers': plan['speakers'],
                 'channels': ['clean', *m.CHANNELS],
                 'rules': {'speakers': 'LibriSpeech dev-clean and dev-other; no speaker appears in any earlier protocol',
                           'cloning': 'voice prompt = one 6–11.5 s utterance per dev-clean speaker; text = a different 3–15 s utterance by that speaker, whose real recording is a human clip',
                           'heldout': 'XTTS-v2 never appears in development, fusion fitting, or threshold calibration',
                           'selection': f'first N per cell in seeded order with readable audio and Whisper WER <= {m.WER_LIMIT}; extra candidates {EXTRA}; detector scores never used',
                           'channels': 'MP3 64 kbps mono; 8 kHz mono G.711 mu-law WAV; white Gaussian noise at 20 dB SNR, seeded per clip'}})
    body['sha256'] = digest(body)
    m.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    m.MANIFEST.write_text(json.dumps(body, indent=2) + '\n')

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items() if k != 'path'}
        return [portable(v) for v in value] if isinstance(value, list) else value
    m.AUDIT.write_text(json.dumps({'manifest_sha256': body['sha256'], 'note': 'Portable record of the frozen v4 test set; local audio paths removed.',
                                   **portable({k: v for k, v in body.items() if k != 'sha256'})}, indent=2) + '\n')
    print(f'Frozen {len(body["cases"])} clips; excluded {len(excluded)}; SHA-256 {body["sha256"]}')


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['plan', 'generate', 'gate', 'channels', 'freeze'])
    parser.add_argument('--generator', choices=list(GENERATORS))
    args = parser.parse_args()
    {'plan': stage_plan, 'gate': m.stage_gate, 'channels': m.stage_channels, 'freeze': stage_freeze,
     'generate': lambda: m.stage_generate(args.generator)}[args.stage]()


if __name__ == '__main__':
    main()
