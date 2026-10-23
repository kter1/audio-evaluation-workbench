from hashlib import sha256
import json

import numpy as np
import pytest
import soundfile as sf

from lab.store import digest
from scripts.screen_candidate_regression import prepare_manifest, regression, write_receipt


def fixture_manifest(tmp_path):
    cases = []
    for index, label in enumerate(('human', 'synthetic')):
        samples = (.08*np.sin(np.arange(6*16000)*2*np.pi*170/16000)).astype('float32')
        path = tmp_path/f'{label}.wav'
        sf.write(path, samples, 16000, subtype='FLOAT')
        identity = sha256(path.read_bytes()).hexdigest()
        case = {'id': label, 'path': path.name, 'sha256': identity, 'label': label,
                'speaker': 'same-speaker', 'source_utt': f'utterance-{index}',
                'ref_utt': 'shared-reference', 'source': 'dev-clean',
                'generator': 'bonafide' if label == 'human' else 'fixture-tts'}
        case['channels'] = {ch: {'path': path.name, 'sha256': identity} for ch in ('mp3', 'phone', 'noise20')}
        cases.append(case)
    document = {'version': '4.0', 'channels': ['clean', 'mp3', 'phone', 'noise20'], 'cases': cases}
    document['sha256'] = digest(document)
    return document


class Detector:
    threshold = .5
    manifest = {'fixture': True}

    def __init__(self):
        self.calls = []

    def score(self, samples):
        self.calls.append(len(samples))
        # Two windows per six-second recording: aggregation is .5, not .99.
        p = .99 if len(self.calls) % 2 else .01
        return {'synthetic_score': p, 'model_input_sha256': sha256(samples.tobytes()).hexdigest()}


def test_regression_preserves_entire_recording_and_connected_sources(tmp_path):
    detector = Detector()
    result = regression(fixture_manifest(tmp_path), tmp_path, detector)
    assert detector.calls == [64000]*16
    assert result['case_count'] == 8 and result['original_clip_count'] == 2
    assert result['source_groups'] == 1
    assert all(r['synthetic_score'] == .5 for r in result['results'])
    first = result['results'][0]
    assert [(w['start'], w['end']) for w in first['windows']] == [(0., 4.), (2., 6.)]
    assert first['assessment']['assessment'] == 'Inconclusive'
    assert result['by_condition']['clean']['raw_false_positive'] == 1
    assert result['by_condition']['clean']['raw_synthetic_detected'] == 1
    assert result['by_condition_generator']['clean']['fixture-tts']['synthetic'] == 1
    tokens = result['input_identities'][0]['identity_tokens']
    assert 'librispeech:speaker:same-speaker' in tokens
    assert 'librispeech:utterance:shared-reference' in tokens


def test_preflight_rejects_hash_change_before_any_inference(tmp_path):
    document = fixture_manifest(tmp_path)
    (tmp_path/'synthetic.wav').write_bytes(b'changed')
    detector = Detector()
    with pytest.raises(ValueError, match='audio hash mismatch'):
        regression(document, tmp_path, detector)
    assert detector.calls == []


def test_incomplete_model_window_coverage_is_rejected(tmp_path):
    detector = Detector()
    detector.spans = lambda x: [(0, 64000)]
    with pytest.raises(ValueError, match='complete anchored'):
        regression(fixture_manifest(tmp_path), tmp_path, detector)
    assert detector.calls == []


def test_native_five_second_context_is_retained_and_declared(tmp_path):
    detector = Detector()
    detector.window_samples = 80000
    result = regression(fixture_manifest(tmp_path), tmp_path, detector)
    assert detector.calls == [80000]*16
    assert [(w['start'], w['end']) for w in result['results'][0]['windows']] == [(0., 5.), (1., 6.)]
    assert result['recording_window_policy']['window_samples'] == 80000
    assert 'native context can differ' in result['recording_window_policy']['comparison']


def test_manifest_self_digest_and_debug_limit_are_enforced(tmp_path):
    document = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match='Debug limit'):
        prepare_manifest(document, tmp_path, 0)
    result = regression(document, tmp_path, Detector(), limit=1)
    assert result['bounds']['complete_manifest_run'] is False
    assert result['bounds']['debug_limit_originals'] == 1
    assert result['case_count'] == 4 and result['manifest_original_count'] == 2
    document['cases'][0]['label'] = 'synthetic'
    with pytest.raises(ValueError, match='self-digest'):
        prepare_manifest(document, tmp_path)


def test_historical_baseline_is_raw_reuse_with_explicit_old_policy(tmp_path):
    document = fixture_manifest(tmp_path)
    baseline = {'version': '4.0', 'manifest_sha256': document['sha256'], 'threshold': .2,
        'test': {'rows': [{'id': c['id'], 'channel': ch, 'label': c['label'], 'score': .1,
                           'assessment': 'Human speech indicated'} for c in document['cases'] for ch in document['channels']]}}
    result = regression(document, tmp_path, Detector(), baseline)
    assert result['historical_baseline']['by_condition']['clean']['raw_false_positive'] == 0
    assert result['historical_baseline']['by_condition']['clean']['raw_synthetic_detected'] == 0
    assert result['results'][0]['historical_baseline']['assessment_under_original_policy'] == 'Human speech indicated'
    baseline['test']['rows'].pop()
    with pytest.raises(ValueError, match='exact complete'):
        regression(document, tmp_path, Detector(), baseline)


def test_receipt_is_content_addressed_and_outputs_never_overwrite(tmp_path):
    result = {'schema_version': 'core-detector-screen-1.0', 'mode': 'supplied development authenticity labels',
              'input_identities': [{'record_id': 'fixture', 'identity_tokens': ['audio:fixture']}]}
    output = tmp_path/'result.json'
    receipt = write_receipt(result, output, tmp_path/'runs')
    assert receipt.read_bytes() == output.read_bytes()
    assert receipt.stem == sha256(receipt.read_bytes().rstrip(b'\n')).hexdigest()
    assert json.loads(output.read_text()) == result
    with pytest.raises(ValueError, match='already exists'):
        write_receipt(result, output, tmp_path/'runs')
