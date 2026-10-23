"""Frozen-detector screening on publisher-labeled development audio; no model fitting."""
from hashlib import sha256
import math
from pathlib import Path

import numpy as np

from lab.baseline_audit import checked_path
from lab.forensics import RATE, abstention_band, decision, diagnostics, mono16, read_audio


def prepare_case(row, root):
    raw = checked_path(Path(root).resolve(), row['path']).read_bytes()
    if sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Screening source hash mismatch')
    if row.get('label') not in ('human', 'synthetic') or not row.get('group'):
        raise ValueError('Publisher label and group required')
    audio, rate, source = read_audio(raw)
    x = mono16(audio, rate)
    count = min(len(x), 4*RATE)
    start = (len(x)-count)//2
    crop = x[start:start+count].copy()
    _, metrics, warnings = diagnostics(crop[:, None], RATE)
    return crop, metrics, warnings, {'source': source, 'crop_start_sample_16k': start,
        'crop_samples': count, 'analysis_samples_sha256': sha256(crop.astype('<f4').tobytes()).hexdigest(),
        'policy': 'Center at most 4 seconds after mono16 conversion; no added padding by screening runner; model-specific padding retained'}


def summarize(rows, threshold):
    if not rows or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Rows and a finite threshold in (0,1) required')
    groups = {}
    for row in rows:
        score = row['synthetic_score']
        if row['label'] not in ('human', 'synthetic') or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid scored screening row')
        group = groups.setdefault(row['group'], {'clips': 0, 'human': 0, 'synthetic': 0,
            'raw_false_positive': 0, 'raw_synthetic_detected': 0, 'inconclusive': 0,
            'human_indicated': 0, 'synthetic_indicated': 0, 'wrong_definitive': 0})
        group['clips'] += 1
        group[row['label']] += 1
        group['raw_false_positive'] += row['label'] == 'human' and score >= threshold
        group['raw_synthetic_detected'] += row['label'] == 'synthetic' and score >= threshold
        assessment = row['assessment']['assessment']
        if assessment not in ('Inconclusive', 'Human speech indicated', 'Synthetic speech indicated'):
            raise ValueError('Unknown assessment state')
        group['inconclusive'] += assessment == 'Inconclusive'
        group['human_indicated'] += assessment == 'Human speech indicated'
        group['synthetic_indicated'] += assessment == 'Synthetic speech indicated'
        group['wrong_definitive'] += (row['label'] == 'human' and assessment == 'Synthetic speech indicated') or (row['label'] == 'synthetic' and assessment == 'Human speech indicated')
    return groups


def screen(acquisition, root, detector, stress=False):
    rows, identities = [], []
    inputs = []
    conditions = ('original', 'gain_minus12db', 'added_noise20db', 'bandwidth8k', 'vorbis_roundtrip') if stress else ('original',)
    for item in acquisition['files']:
        original, _, _, provenance = prepare_case(item, root)
        for condition in conditions:
            if condition == 'original':
                crop, metadata = original, {'condition': condition}
            else:
                from lab.vad_stress import transform
                seed = int(sha256((item['sha256']+':core-screen-1.0:'+condition).encode()).hexdigest()[:16],16)
                crop, _, metadata = transform(original, None, condition, seed)
                metadata.pop('codec_payload', None)
            inputs.append((item, crop, provenance, condition, metadata))
    for item, crop, provenance, condition, metadata in inputs:
        _, metrics, warnings = diagnostics(crop[:, None], RATE)
        score = detector.score(crop)
        if not np.isfinite(score['synthetic_score']):
            raise ValueError('Nonfinite detector score')
        span = {'start': 0., 'end': len(crop)/RATE, **score}
        band = abstention_band(detector.threshold, getattr(detector, 'band_logit', None))
        assessment = decision([span], metrics, len(crop)/RATE, detector.threshold, band)
        record_id = 'voxenes-2026:'+item['name']+'::'+condition
        identities.append({'record_id': record_id, 'identity_tokens': sorted({
            'audio:'+item['sha256'], 'analysis-pcm:'+sha256(crop.astype('<f4').tobytes()).hexdigest(),
            'voxenes-2026:unresolved-source-speaker-identities'})})
        rows.append({'id': record_id, 'label': item['label'], 'group': item['group'],
            **provenance, 'condition': condition, 'transformation': metadata,
            'transformed_samples_sha256': sha256(crop.astype('<f4').tobytes()).hexdigest(), 'synthetic_score': score['synthetic_score'], 'components': score.get('components', {}),
            'assessment': assessment, 'warnings': warnings})
        if 'model_input_sha256' in score:
            rows[-1]['model_input_sha256'] = score['model_input_sha256']
    return {'schema_version': 'core-detector-screen-1.0', 'mode': 'supplied development authenticity labels',
        'threshold': detector.threshold, 'model': detector.manifest, 'case_count': len(rows),
        'source_groups': 1, 'input_identities': identities, 'results': rows,
        'by_group': summarize(rows, detector.threshold),
        'by_condition': {condition: summarize([r for r in rows if r['condition']==condition], detector.threshold) for condition in conditions},
        'original_clip_count': len(acquisition['files']), 'selection': acquisition['spec']['selection'],
        'scope': 'Small publisher-labeled modern-generator development screen. Frozen existing detector, no fitting or threshold tuning. '
                 'No speaker/source/training independence established; all clips conservatively one connected group. '
                 'Center crops with declared constructed conditions; original labels inherited, audibility not re-reviewed. '
                 'Score/quality-band checks applied; product perturbation-stability check not run. Not a complete dataset benchmark, product end-to-end accuracy estimate or world-class claim.'}
