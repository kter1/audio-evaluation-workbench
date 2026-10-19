"""Blind human-vs-detector listening challenge, built on the existing review workflow.

Clips come from the frozen modern-voice test split and are chosen by seed, never by
detector score. Detector scores come from the frozen evaluation receipt, so the
detector's answer exists before anyone listens.
"""
from pathlib import Path
import random
import uuid

import numpy as np
import soundfile as sf

from .audio import decode
from .forensics import RATE, mono16
from .store import canonical, digest, now, register_current_guide

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261019
TARGET_RMS_DBFS = -26.0
RENDERING = (f'16 kHz mono 16-bit WAV (scipy resample_poly, as the detector does) at {TARGET_RMS_DBFS:.0f} dBFS RMS, '
             'peak-limited to 0.99, so sample rate and loudness do not reveal origin. Both detectors normalize each '
             'window, so the frozen scores do not depend on loudness.')


def default_fakes(manifest):
    generators = sorted({c['generator'] for c in manifest['cases'] if c['label'] == 'synthetic'})
    held = manifest.get('heldout_generator')
    if held in generators:  # weight the generator the detector never saw in development
        others = [g for g in generators if g != held]
        return {**{g: 3 for g in others}, held: 10 - 3 * len(others)}
    return {g: 10 // len(generators) for g in generators}


def choose(manifest, seed=SEED, real=10, fakes=None):
    """Seeded selection from the clean test split; real clips share the fakes' speakers."""
    rng = random.Random(seed)
    fakes = fakes or default_fakes(manifest)
    test = sorted((c for c in manifest['cases'] if c['split'] == 'test'), key=lambda c: c['id'])
    picked = rng.sample([c for c in test if c['label'] == 'human' and c['source'] in ('test-clean', 'dev-clean')], real)
    for generator, n in sorted(fakes.items()):
        picked += rng.sample([c for c in test if c['generator'] == generator], n)
    rng.shuffle(picked)
    return picked


def render(path):
    audio, rate = decode(path)
    x = mono16(audio, rate).astype(np.float64)
    rms = np.sqrt(np.mean(x**2))
    if rms <= 0:
        raise ValueError(f'Silent clip cannot be level-matched: {path}')
    x *= 10**(TARGET_RMS_DBFS/20) / rms
    peak = np.max(np.abs(x))
    if peak > .99:
        x *= .99 / peak
    return x.astype(np.float32)


def prepare_challenge(store, manifest, receipt, reviewer, seed=SEED, root=ROOT):
    if not reviewer.strip():
        raise ValueError('Reviewer name required')
    if store.one('SELECT 1 FROM recordings LIMIT 1'):
        raise ValueError('Prepare the challenge in a new database')
    if receipt.get('version') not in ('3.0', '4.0') or receipt.get('manifest_sha256') != manifest.get('sha256'):
        raise ValueError('The receipt must be the modern-voice evaluation of this exact frozen manifest')
    scores = {r['id']: r['score'] for r in receipt['test']['rows'] if r['channel'] == 'clean'}
    threshold = receipt['threshold']
    clips = choose(manifest, seed)
    if any(c['id'] not in scores for c in clips):
        raise ValueError('Every challenge clip needs its frozen clean-condition detector score')
    media = store.path.parent / (store.path.stem + '_audio')
    media.mkdir(parents=True, exist_ok=True)
    records, score_rows, selected = [], [], []
    for i, clip in enumerate(clips, 1):
        trial = f'CH_{i:03d}'
        path = media / f'{trial}.wav'
        sf.write(path, render(Path(root) / clip['path']), RATE, subtype='PCM_16')
        records.append({'trial_id': trial, 'path': str(path), 'source_group': f'librispeech:{clip["speaker"]}',
                        'subset': 'challenge', 'codec': 'clean · 16 kHz render', 'kind': 'research',
                        'attack': clip['generator'], 'reference': clip['label']})
        score_rows.append({'trial_id': trial, 'score': scores[clip['id']]})
        selected.append({'trial_id': trial, 'source_id': clip['id'], 'label': clip['label'], 'generator': clip['generator'],
                         'speaker': clip['speaker'], 'source_sha256': clip['sha256'], 'detector_score': scores[clip['id']]})
    provenance = {'dataset': 'Modern-voice listening challenge', 'manifest_sha256': manifest['sha256'],
                  'receipt_sha256': digest(receipt), 'detector': receipt['selected_model'], 'threshold': threshold,
                  'rendering': RENDERING, 'license': 'LibriSpeech CC BY 4.0; generated clips local research use only'}
    store.register(records, score_rows, provenance)
    guideline = register_current_guide(store)
    with store.conn:
        store.set_setting('dataset_kind', 'challenge')
    rows = store.rows('SELECT id,trial_id FROM recordings ORDER BY trial_id')
    for row in rows:
        store.validate_media(row['id'])
    config = {'threshold': threshold, 'direction': 'higher_is_synthetic', 'comparison': 'score >= threshold',
              'detector': receipt['selected_model'], 'receipt_sha256': digest(receipt), 'frozen_at': now()}
    study = {'seed': seed, 'counts': {'challenge': len(clips)}, 'selected': selected, 'reviewer': reviewer,
             'import_provenance': store.setting('import_provenance'), 'manifest_sha256': manifest['sha256'],
             'receipt_sha256': digest(receipt), 'rendering': RENDERING}
    with store.conn:
        store.conn.execute('INSERT INTO threshold_config VALUES(1,?)', (canonical(config),))
        for order, row in enumerate(rows):
            store.conn.execute('INSERT INTO assignments(id,recording_id,reviewer,phase,role,guideline,ordinal) VALUES(?,?,?,?,?,?,?)',
                               (uuid.uuid4().hex, row['id'], reviewer, 'challenge', 'primary', guideline, order))
        store.conn.execute('INSERT INTO study VALUES(1,?,?)', (canonical(study), digest(study)))
        store.set_setting('primary_reviewer', reviewer)
    return study
