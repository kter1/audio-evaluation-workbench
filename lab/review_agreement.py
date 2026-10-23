"""Validate completed development reviews and measure descriptive agreement."""
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path

from lab.baseline_audit import checked_path
from lab.research_metrics import speech_frames, validate_intervals


def validate_review(packet, review_path):
    packet = Path(packet)
    template = json.loads((packet/'reviewer/annotations.json').read_text())
    raw = Path(review_path).read_bytes()
    review = json.loads(raw)
    if review.get('schema_version') != 'speech-review-1.0' or review.get('partition') != 'development':
        raise ValueError('Expected development speech review')
    reviewer = review.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Completed review requires reviewer identifier')
    try:
        completed = datetime.fromisoformat(review['completed_at'].replace('Z', '+00:00'))
        if completed.utcoffset() is None:
            raise ValueError('Timezone required')
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Completion time must be an ISO timestamp with timezone') from exc
    rows = review.get('cases')
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('Review cases must be objects')
    ids = [r.get('id') for r in rows]
    expected = {r['id']: r for r in template['cases']}
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('Review must contain every packet case exactly once')
    for row in rows:
        original = expected[row['id']]
        if any(row.get(k) != original[k] for k in ('path', 'sha256', 'duration_seconds')):
            raise ValueError('Review changed packet identity or duration')
        audio = checked_path((packet/'reviewer').resolve(), original['path'])
        if sha256(audio.read_bytes()).hexdigest() != original['sha256']:
            raise ValueError('Packet audio hash mismatch')
        if row.get('status') != 'reviewed' or not isinstance(row.get('speech_intervals'), list):
            raise ValueError('Every case must be reviewed; null is not a negative label')
        if not isinstance(row.get('notes'), str):
            raise ValueError('Review notes must be text')
        intervals = row['speech_intervals']
        if any(not isinstance(i, dict) or not {'start', 'end'} <= i.keys() for i in intervals):
            raise ValueError('Speech intervals require start and end')
        validate_intervals(intervals, original['duration_seconds'], nonoverlap=True)
    return review, sha256(raw).hexdigest()


def compare_reviews(packet, left_path, right_path):
    left, left_hash = validate_review(packet, left_path)
    right, right_hash = validate_review(packet, right_path)
    if left['reviewer'].strip().casefold() == right['reviewer'].strip().casefold():
        raise ValueError('Distinct reviewer identifiers required; independence still needs external verification')
    right_rows = {r['id']: r for r in right['cases']}
    counts = dict(both_speech=0, left_only=0, right_only=0, both_nonspeech=0)
    rows = []
    for lrow in left['cases']:
        rrow = right_rows[lrow['id']]
        duration = lrow['duration_seconds']
        analyzed = math.floor(duration/.02 + 1e-9)*.02
        if analyzed <= 0:
            raise ValueError('Clip too short for common 20 ms evaluation')
        def clipped(intervals):
            return [{'start': i['start'], 'end': min(i['end'], analyzed)}
                    for i in intervals if i['start'] < analyzed]
        metric = speech_frames(clipped(lrow['speech_intervals']), clipped(rrow['speech_intervals']), analyzed)
        values = dict(zip(counts, (metric['true_positive'], metric['false_positive'],
                                  metric['false_negative'], metric['true_negative'])))
        for key, value in values.items():
            counts[key] += value
        disagreement = values['left_only']+values['right_only']
        rows.append({'id': lrow['id'], 'frames': metric['frames'],
                     'disagreement_seconds': disagreement*.02,
                     'ignored_tail_seconds': max(0., duration-analyzed),
                     'needs_adjudication': bool(disagreement) or lrow['speech_intervals'] != rrow['speech_intervals']
                                           or bool(lrow['notes'].strip() or rrow['notes'].strip())})
    total = sum(counts.values())
    observed = (counts['both_speech']+counts['both_nonspeech'])/total
    lp = (counts['both_speech']+counts['left_only'])/total
    rp = (counts['both_speech']+counts['right_only'])/total
    chance = lp*rp+(1-lp)*(1-rp)
    return {'schema_version': 'speech-agreement-1.0', 'review_sha256': [left_hash, right_hash],
            'reviewers': [left['reviewer'], right['reviewer']], 'case_count': len(rows),
            'hop_seconds': .02, 'counts': counts, 'observed_agreement': observed,
            'cohen_kappa': (observed-chance)/(1-chance) if chance < 1 else None,
            'cases': rows, 'adjudication_count': sum(r['needs_adjudication'] for r in rows),
            'scope': 'Descriptive paired frame agreement, not accuracy or proof of reviewer independence. '
                     'Frame centers in [start,end); incomplete tails excluded. No population interval or promotion. '
                     'Interval differences and notes require adjudication even if frame decisions agree.'}
