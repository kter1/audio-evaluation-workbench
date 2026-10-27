from hashlib import sha256
import json

import numpy as np
import pytest
import soundfile as sf

from lab.annotation_review import build_packet


def fixture(tmp_path):
    source = tmp_path/'identity.wav'
    sf.write(source, np.zeros(1600), 16000)
    manifest = tmp_path/'manifest.json'
    row = {'id': 'publisher-secret', 'path': source.name,
           'source_group': 'speaker-secret', 'sha256': sha256(source.read_bytes()).hexdigest(),
           'speech_intervals': [{'start': 0, 'end': .05}]}
    manifest.write_text(json.dumps({'schema_version': 'method-benchmark-1.0',
                                   'partition': 'development', 'cases': [row]}))
    return manifest, row


def test_packet_blinds_labels_preserves_bytes_and_pending_state(tmp_path):
    manifest, row = fixture(tmp_path)
    target = tmp_path/'packet'
    receipt = build_packet(manifest, tmp_path, target)
    review = json.loads((target/'reviewer/annotations.json').read_text())
    case = review['cases'][0]
    assert case['speech_intervals'] is None and case['status'] == 'pending'
    assert review['reviewer'] is None
    assert sha256((target/'reviewer'/case['path']).read_bytes()).hexdigest() == row['sha256']
    assert receipt['mapping'][0]['source_id'] == row['id']
    for p in (target/'reviewer').rglob('*'):
        if p.is_file() and p.suffix != '.wav':
            assert 'publisher-secret' not in p.read_text()
            assert 'speaker-secret' not in p.read_text()
    with pytest.raises(ValueError, match='must not exist'):
        build_packet(manifest, tmp_path, target)


@pytest.mark.parametrize('failure', ['tamper', 'locked', 'duplicate', 'escape'])
def test_invalid_input_leaves_no_packet(tmp_path, failure):
    manifest, row = fixture(tmp_path)
    doc = json.loads(manifest.read_text())
    if failure == 'tamper':
        row['sha256'] = '0'*64
    elif failure == 'locked':
        doc['partition'] = 'locked_test'
    elif failure == 'duplicate':
        doc['cases'].append(row.copy())
    else:
        row['path'] = '../outside.wav'
    doc['cases'][0] = row
    manifest.write_text(json.dumps(doc))
    with pytest.raises((ValueError, FileNotFoundError)):
        build_packet(manifest, tmp_path, tmp_path/'packet')
    assert not (tmp_path/'packet').exists()
    assert not list(tmp_path.glob('.review-*'))
