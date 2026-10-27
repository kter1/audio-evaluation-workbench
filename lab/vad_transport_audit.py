"""Verify constructed transport failures reach existing audio quality warnings."""
from hashlib import sha256
import json
from pathlib import Path

from lab.baseline_audit import checked_path
from lab.forensics import diagnostics, read_audio

AUDITED = ('stereo_antiphase', 'crop_center500ms', 'crop_center1s', 'crop_center2s')


def audit_transport(manifest_path, root):
    raw_manifest = Path(manifest_path).read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get('partition') != 'development' or manifest.get('stress_protocol', {}).get('version') != 'vad-channel-stress-1.0':
        raise ValueError('Expected development channel stress manifest')
    selected = [r for r in manifest['cases'] if r.get('condition') in AUDITED]
    count = manifest['stress_protocol']['source_case_count']
    parents = {}
    for row in selected:
        conditions = parents.setdefault(row['parent_id'], set())
        if row['condition'] in conditions:
            raise ValueError('Duplicate transport parent/condition')
        conditions.add(row['condition'])
    if len(parents) != count or any(conditions != set(AUDITED) for conditions in parents.values()):
        raise ValueError('Exactly one case per source and audited condition required')
    rows = []
    for row in selected:
        raw = checked_path(Path(root).resolve(), row['path']).read_bytes()
        if sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Transport audio hash mismatch')
        audio, rate, _ = read_audio(raw)
        x, metrics, warnings = diagnostics(audio, rate)
        if row['condition'] == 'stereo_antiphase':
            source = metrics['source_measurements']
            checks = {'two_source_channels': audio.shape[1] == 2,
                      'source_channels_nonzero': all(c['rms_amplitude'] > 0 for c in source['channels']),
                      'exact_mono_cancellation': not x.any() and source['arithmetic_mixdown']['complete_cancellation'],
                      'cancellation_warning_present': any('Opposite-phase' in w for w in warnings),
                      'inactive_signal_warning_present': any('Insufficient active signal' in w for w in warnings)}
        else:
            checks = {'short_audio': .5 <= len(x)/16000 <= 2.,
                      'minimum_context_warning_present': any('minimum-context policy' in w for w in warnings)}
        rows.append({'id': row['id'], 'condition': row['condition'], 'sha256': row['sha256'],
                     'checks': checks, 'warnings': warnings})
    return {'schema_version': 'vad-transport-audit-1.0', 'manifest_sha256': sha256(raw_manifest).hexdigest(),
            'case_count': len(rows), 'passed_cases': sum(all(r['checks'].values()) for r in rows), 'cases': rows,
            'by_condition': {condition: {'case_count': sum(r['condition'] == condition for r in rows),
                'passed_cases': sum(r['condition'] == condition and all(r['checks'].values()) for r in rows)} for condition in AUDITED},
            'scope': 'Known constructed cancellation/short-context diagnostics and existing warnings only; '
                     'no authenticity-model inference or proof of end-to-end abstention.'}
