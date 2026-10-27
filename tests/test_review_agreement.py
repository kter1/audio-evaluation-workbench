import copy
import json

import numpy as np
import pytest
import soundfile as sf
from hashlib import sha256

from lab.annotation_review import build_packet
from lab.review_agreement import compare_reviews, validate_review


def reviews(tmp_path, samples=1600):
    audio = tmp_path/'a.wav'
    sf.write(audio, np.zeros(samples), 16000)
    manifest = tmp_path/'m.json'
    manifest.write_text(json.dumps({'schema_version': 'method-benchmark-1.0', 'partition': 'development',
        'cases': [{'id': 'a', 'path': 'a.wav', 'sha256': sha256(audio.read_bytes()).hexdigest(), 'source_group': 'g'}]}))
    packet = tmp_path/'packet'
    build_packet(manifest, tmp_path, packet)
    doc = json.loads((packet/'reviewer/annotations.json').read_text())
    doc.update(reviewer='one', completed_at='2026-10-23T12:00:00Z')
    doc['cases'][0].update(status='reviewed', speech_intervals=[{'start': 0., 'end': .04}])
    left, right = tmp_path/'l.json', tmp_path/'r.json'
    left.write_text(json.dumps(doc))
    other = copy.deepcopy(doc)
    other['reviewer'] = 'two'
    other['cases'][0]['speech_intervals'] = [{'start': .02, 'end': .06}]
    right.write_text(json.dumps(other))
    return packet, left, right


def test_known_agreement_and_symmetry(tmp_path):
    p, l, r = reviews(tmp_path)
    result = compare_reviews(p, l, r)
    assert result['counts'] == dict(both_speech=1, left_only=1, right_only=1, both_nonspeech=2)
    assert result['observed_agreement'] == .6
    assert result['cohen_kappa'] == pytest.approx(1/6)
    assert result['adjudication_count'] == 1
    assert compare_reviews(p, r, l)['cohen_kappa'] == result['cohen_kappa']


@pytest.mark.parametrize('change', ['pending', 'null', 'overlap', 'bounds', 'identity', 'missing', 'duplicate', 'time', 'reviewer', 'nan'])
def test_rejects_invalid_completion(tmp_path, change):
    p, l, _ = reviews(tmp_path)
    doc = json.loads(l.read_text())
    row = doc['cases'][0]
    if change == 'pending': row['status'] = 'pending'
    elif change == 'null': row['speech_intervals'] = None
    elif change == 'overlap': row['speech_intervals'] *= 2
    elif change == 'bounds': row['speech_intervals'][0]['end'] = 1
    elif change == 'identity': row['sha256'] = '0'*64
    elif change == 'missing': doc['cases'] = []
    elif change == 'duplicate': doc['cases'] *= 2
    elif change == 'time': doc['completed_at'] = '2026-10-23'
    elif change == 'reviewer': doc['reviewer'] = ' '
    else: row['speech_intervals'][0]['end'] = float('nan')
    l.write_text(json.dumps(doc))
    with pytest.raises(ValueError): validate_review(p, l)


def test_same_reviewer_rejected_and_degenerate_kappa_undefined(tmp_path):
    p, l, r = reviews(tmp_path)
    doc = json.loads(l.read_text())
    doc['cases'][0]['speech_intervals'] = []
    l.write_text(json.dumps(doc)); r.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='Distinct'): compare_reviews(p, l, r)
    doc['reviewer'] = 'two'
    r.write_text(json.dumps(doc))
    result = compare_reviews(p, l, r)
    assert result['observed_agreement'] == 1
    assert result['cohen_kappa'] is None
    assert result['adjudication_count'] == 0


def test_audio_tampering_rejected(tmp_path):
    p, l, _ = reviews(tmp_path)
    next((p/'reviewer/audio').glob('*.wav')).write_bytes(b'changed')
    with pytest.raises(ValueError, match='audio hash'): validate_review(p, l)


def test_subframe_boundary_difference_still_requires_adjudication(tmp_path):
    p, l, r = reviews(tmp_path)
    doc = json.loads(l.read_text())
    doc['reviewer'] = 'two'
    doc['cases'][0]['speech_intervals'][0]['end'] = .041
    r.write_text(json.dumps(doc))
    result = compare_reviews(p, l, r)
    assert result['observed_agreement'] == 1
    assert result['cases'][0]['disagreement_seconds'] == 0
    assert result['adjudication_count'] == 1


from lab.review_adjudication import export_adjudicated, prepare_adjudication


def completed_decisions(tmp_path, samples=1600):
    packet, left, right = reviews(tmp_path, samples=samples)
    decisions = prepare_adjudication(packet, left, right)
    assert decisions['cases'][0]['speech_intervals'] is None
    decisions.update(adjudicator='third', completed_at='2026-10-23T13:00:00Z')
    decisions['cases'][0].update(status='resolved', rationale='Listened to the full clip; resolved onset.',
                               speech_intervals=[{'start': .01, 'end': .05}])
    path = tmp_path/'decisions.json'
    path.write_text(json.dumps(decisions))
    return packet, left, right, path


def test_adjudication_exports_development_labels_with_hash_lineage(tmp_path):
    p, l, r, d = completed_decisions(tmp_path)
    original = (tmp_path/'m.json').read_bytes()
    result = export_adjudicated(p, l, r, d, tmp_path/'m.json')
    assert result['partition'] == 'development'
    assert result['cases'][0]['speech_intervals'] == [{'start': .01, 'end': .05}]
    assert result['review_evidence']['resolved_disputes'] == 1
    assert result['review_evidence']['adjudication_sha256'] == sha256(d.read_bytes()).hexdigest()
    assert (tmp_path/'m.json').read_bytes() == original
    assert json.loads(l.read_text())['cases'][0]['speech_intervals'] == [{'start': 0., 'end': .04}]


@pytest.mark.parametrize('change', ['pending', 'null', 'rationale', 'missing', 'duplicate', 'extra',
                                    'context', 'out_of_bounds', 'review_hash', 'mapping_hash', 'time', 'adjudicator'])
def test_adjudication_rejects_incomplete_or_changed_evidence(tmp_path, change):
    p, l, r, d = completed_decisions(tmp_path)
    doc = json.loads(d.read_text())
    row = doc['cases'][0]
    if change == 'pending': row['status'] = 'pending'
    elif change == 'null': row['speech_intervals'] = None
    elif change == 'rationale': row['rationale'] = ' '
    elif change == 'missing': doc['cases'] = []
    elif change == 'duplicate': doc['cases'] *= 2
    elif change == 'extra': doc['cases'].append({**row, 'id': 'not-in-packet'})
    elif change == 'context': row['left_notes'] = 'edited'
    elif change == 'out_of_bounds': row['speech_intervals'][0]['end'] = 1
    elif change == 'review_hash': doc['review_sha256'][0] = '0'*64
    elif change == 'mapping_hash': doc['packet_mapping_sha256'] = '0'*64
    elif change == 'time': doc['completed_at'] = '2026-10-23'
    else: doc['adjudicator'] = ''
    d.write_text(json.dumps(doc))
    with pytest.raises(ValueError): export_adjudicated(p, l, r, d, tmp_path/'m.json')


def test_changed_source_manifest_cannot_receive_review_labels(tmp_path):
    p, l, r, d = completed_decisions(tmp_path)
    manifest = tmp_path/'m.json'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError, match='manifest hash'):
        export_adjudicated(p, l, r, d, manifest)


def test_consensus_needs_no_interval_decision_but_requires_signoff(tmp_path):
    p, l, r = reviews(tmp_path)
    right = json.loads(l.read_text()); right['reviewer'] = 'two'
    r.write_text(json.dumps(right))
    doc = prepare_adjudication(p, l, r)
    assert doc['cases'] == []
    doc.update(adjudicator='coordinator', completed_at='2026-10-23T13:00:00Z')
    path = tmp_path/'decisions.json'; path.write_text(json.dumps(doc))
    result = export_adjudicated(p, l, r, path, tmp_path/'m.json')
    assert result['review_evidence']['resolved_disputes'] == 0
    assert result['cases'][0]['speech_intervals'] == right['cases'][0]['speech_intervals']


def test_adjudicated_export_accepted_by_development_runner(tmp_path):
    from lab.research_benchmark import load_manifest
    p, l, r, d = completed_decisions(tmp_path, samples=16000)
    result = export_adjudicated(p, l, r, d, tmp_path/'m.json')
    manifest = tmp_path/'adjudicated.json'; manifest.write_text(json.dumps(result))
    # Minimal constructed manifest needs the benchmark's original provenance field;
    # export supplies the replacement provenance without inventing other labels.
    cases = load_manifest(manifest, tmp_path)
    assert cases[0]['speech_intervals'] == [{'start': .01, 'end': .05}]


def test_dispute_adjudicator_must_differ_from_initial_reviewers(tmp_path):
    p, l, r, d = completed_decisions(tmp_path)
    doc = json.loads(d.read_text()); doc['adjudicator'] = ' ONE '
    d.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='distinct adjudicator'):
        export_adjudicated(p, l, r, d, tmp_path/'m.json')
