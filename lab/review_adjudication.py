"""Explicit resolution of development speech reviews; never creates locked labels."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from lab.review_agreement import compare_reviews, validate_review
from lab.research_metrics import validate_intervals


def _inputs(packet, left_path, right_path):
    agreement = compare_reviews(packet, left_path, right_path)
    left, lh = validate_review(packet, left_path)
    right, rh = validate_review(packet, right_path)
    if [lh, rh] != agreement['review_sha256']:
        raise ValueError('Review changed during comparison')
    return agreement, {r['id']: r for r in left['cases']}, {r['id']: r for r in right['cases']}


def prepare_adjudication(packet, left_path, right_path):
    agreement, left, right = _inputs(packet, left_path, right_path)
    rows = []
    for case in agreement['cases']:
        if case['needs_adjudication']:
            key = case['id']
            rows.append({'id': key, 'duration_seconds': left[key]['duration_seconds'],
                         'left_intervals': left[key]['speech_intervals'],
                         'right_intervals': right[key]['speech_intervals'],
                         'left_notes': left[key]['notes'], 'right_notes': right[key]['notes'],
                         'status': 'pending', 'speech_intervals': None, 'rationale': ''})
    return {'schema_version': 'speech-adjudication-1.0', 'partition': 'development',
            'review_sha256': agreement['review_sha256'],
            'packet_mapping_sha256': sha256((Path(packet)/'coordinator-only/mapping.json').read_bytes()).hexdigest(),
            'adjudicator': None, 'completed_at': None, 'cases': rows,
            'instructions': 'Listen to complete disputed clips; explicitly resolve every interval/notes dispute. '
                            'Enter final speech intervals, nonempty rationale and resolved status for each case. '
                            'Retain original review files. Null is pending; [] is reviewed nonspeech. '
                            'No model predictions are supplied. Development labels only.'}


def export_adjudicated(packet, left_path, right_path, decisions_path, manifest_path):
    expected = prepare_adjudication(packet, left_path, right_path)
    raw = Path(decisions_path).read_bytes()
    decisions = json.loads(raw)
    for key in ('schema_version', 'partition', 'review_sha256', 'packet_mapping_sha256'):
        if decisions.get(key) != expected[key]:
            raise ValueError('Adjudication inputs changed: '+key)
    who = decisions.get('adjudicator')
    if not isinstance(who, str) or not who.strip():
        raise ValueError('Adjudicator identifier required')
    if expected['cases']:
        reviewers = [validate_review(packet, path)[0]['reviewer'].strip().casefold()
                     for path in (left_path, right_path)]
        if who.strip().casefold() in reviewers:
            raise ValueError('Disputes require a distinct adjudicator identifier; independence needs external verification')
    try:
        time = datetime.fromisoformat(decisions['completed_at'].replace('Z', '+00:00'))
        if time.utcoffset() is None:
            raise ValueError('Missing timezone')
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Adjudication completion requires timezone-bearing ISO timestamp') from exc
    rows = decisions.get('cases')
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) for r in rows):
        raise ValueError('Adjudication cases must be identified objects')
    by_id = {r['id']: r for r in rows}
    required = {r['id']: r for r in expected['cases']}
    if len(by_id) != len(rows) or set(by_id) != set(required):
        raise ValueError('Every disputed case must be resolved exactly once')
    for key, row in by_id.items():
        for field in ('duration_seconds', 'left_intervals', 'right_intervals', 'left_notes', 'right_notes'):
            if row.get(field) != required[key][field]:
                raise ValueError('Original review context changed')
        if row.get('status') != 'resolved' or not isinstance(row.get('speech_intervals'), list):
            raise ValueError('All disputes require explicit resolved intervals')
        if not isinstance(row.get('rationale'), str) or not row['rationale'].strip():
            raise ValueError('Resolution rationale required')
        intervals = row['speech_intervals']
        if any(not isinstance(i, dict) or not {'start', 'end'} <= i.keys() for i in intervals):
            raise ValueError('Resolution intervals require start and end')
        validate_intervals(intervals, required[key]['duration_seconds'], nonoverlap=True)
    packet = Path(packet)
    mapping = json.loads((packet/'coordinator-only/mapping.json').read_text())
    manifest_raw = Path(manifest_path).read_bytes()
    if sha256(manifest_raw).hexdigest() != mapping['manifest_sha256']:
        raise ValueError('Source manifest hash mismatch')
    manifest = json.loads(manifest_raw)
    if manifest.get('partition') != 'development':
        raise ValueError('Only development export permitted')
    left, left_hash = validate_review(packet, left_path)
    if left_hash != expected['review_sha256'][0]:
        raise ValueError('Review changed during export')
    reviews = {r['id']: r for r in left['cases']}
    sources = {r['source_id']: r for r in mapping['mapping']}
    if len(sources) != len(mapping['mapping']) or set(sources) != {r['id'] for r in manifest['cases']}:
        raise ValueError('Packet mapping does not cover source manifest exactly')
    final_hash = sha256(raw).hexdigest()
    for row in manifest['cases']:
        mapped = sources[row['id']]
        review_id = mapped['review_id']
        if mapped['sha256'] != row['sha256'] or reviews[review_id]['sha256'] != row['sha256']:
            raise ValueError('Source and review audio identities differ')
        row['speech_intervals'] = (by_id[review_id] if review_id in by_id else reviews[review_id])['speech_intervals']
        if 'annotation_sha256' in row:
            row['publisher_annotation_sha256'] = row.pop('annotation_sha256')
        row['publisher_annotation_provenance'] = row.get('annotation_provenance')
        row['annotation_provenance'] = 'Paired completed development reviews with explicit dispute adjudication; reviewer independence not programmatically verified'
        row['adjudication_sha256'] = final_hash
    manifest['review_evidence'] = {'review_sha256': expected['review_sha256'],
        'adjudication_sha256': final_hash, 'packet_mapping_sha256': expected['packet_mapping_sha256'],
        'adjudicator': who, 'resolved_disputes': len(rows),
        'scope': 'Development only; does not qualify corpus, establish identity/training independence or promote methods.'}
    return manifest
