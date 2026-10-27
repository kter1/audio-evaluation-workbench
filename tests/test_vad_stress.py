from hashlib import sha256
import json

import numpy as np
import pytest
import soundfile as sf

from lab.vad_stress import CONDITIONS, RATE, build_stress, summarize_stress, transform


def source(tmp_path):
    x = np.sin(np.arange(RATE)*2*np.pi*400/RATE).astype(np.float32)*.2
    path = tmp_path/'source.wav'; sf.write(path, x, RATE, subtype='FLOAT')
    doc = {'schema_version': 'method-benchmark-1.0', 'partition': 'development',
           'cases': [{'id': 'source', 'path': path.name, 'sha256': sha256(path.read_bytes()).hexdigest(),
                      'source_group': 'source-group', 'speaker_group': 'speaker-group',
                      'annotation_provenance': 'constructed signal only', 'speech_intervals': [],
                      'level_events': [], 'noise_reference_dbfs': None}]}
    manifest = tmp_path/'source.json'; manifest.write_text(json.dumps(doc))
    return manifest


def test_transform_gain_noise_ratio_and_determinism():
    x = np.random.default_rng(8).normal(0, .1, RATE).astype(np.float32)
    labels = [{'start': .1, 'end': .8}]
    y, actual, _ = transform(x, labels, 'gain_minus24db', 3)
    assert np.sqrt(np.mean(y.astype(float)**2)/np.mean(x.astype(float)**2)) == pytest.approx(10**(-24/20), rel=1e-6)
    assert actual == labels and actual is not labels
    a, _, meta = transform(x, labels, 'added_noise10db', 3)
    b, _, _ = transform(x, labels, 'added_noise10db', 3)
    assert np.array_equal(a, b)
    ratio = 10*np.log10(np.mean(x.astype(float)**2)/np.mean((a.astype(float)-x)**2))
    assert ratio == pytest.approx(10, abs=1e-5)
    assert 'not clean-speech SNR' in meta['ratio_scope']


def test_prefix_labels_exact_samples_unknown_and_empty():
    x = np.ones(RATE, np.float32)*.1
    y, labels, meta = transform(x, [{'start': 0, 'end': 1}], 'prefix137ms', 0)
    assert len(y) == RATE+2192 and np.count_nonzero(y[:2192]) == 0
    assert labels == [{'start': .137, 'end': 1.137}]
    assert np.array_equal(y[2192:], x)
    assert transform(x, None, 'prefix137ms', 0)[1] is None
    assert transform(x, [], 'prefix137ms', 0)[1] == []
    with pytest.raises(ValueError, match='zero-RMS'):
        transform(np.zeros(RATE), [], 'added_noise20db', 1)


def test_stress_preserves_identity_and_original_inputs(tmp_path):
    manifest = source(tmp_path); original = manifest.read_bytes()
    doc = build_stress(manifest, tmp_path, tmp_path/'stress')
    assert len(doc['cases']) == 6
    assert set(c['condition'] for c in doc['cases']) == set(CONDITIONS)
    assert all(c['source_group'] == 'source-group' and c['speaker_group'] == 'speaker-group' for c in doc['cases'])
    assert all(c['parent_id'] == 'source' for c in doc['cases'])
    assert all('audio:'+c['parent_sha256'] in c['source_references'] for c in doc['cases'])
    for row in doc['cases']:
        assert sha256((tmp_path/row['path']).read_bytes()).hexdigest() == row['sha256']
        assert row['speech_intervals'] == []
    assert manifest.read_bytes() == original
    with pytest.raises(ValueError, match='must not exist'):
        build_stress(manifest, tmp_path, tmp_path/'stress')


@pytest.mark.parametrize('change', ['hash', 'locked', 'rate', 'unknown_silent_noise'])
def test_generation_failures_leave_no_partial_output(tmp_path, change):
    manifest = source(tmp_path)
    doc = json.loads(manifest.read_text())
    if change == 'hash': doc['cases'][0]['sha256'] = '0'*64
    elif change == 'locked': doc['partition'] = 'locked_test'
    elif change == 'rate': sf.write(tmp_path/'source.wav', np.zeros(8000), 8000)
    else:
        sf.write(tmp_path/'source.wav', np.zeros(RATE), RATE)
        doc['cases'][0]['sha256'] = sha256((tmp_path/'source.wav').read_bytes()).hexdigest()
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError): build_stress(manifest, tmp_path, tmp_path/'stress')
    assert not (tmp_path/'stress').exists()
    assert not list(tmp_path.glob('.vad-stress-*'))


def test_summary_rejects_missing_and_duplicate_conditions(tmp_path):
    manifest = source(tmp_path)
    doc = build_stress(manifest, tmp_path, tmp_path/'stress')
    result = {'results': [{'id': row['id']} for row in doc['cases']]}
    doc['cases'][1]['condition'] = 'original'
    with pytest.raises(ValueError, match='Duplicate parent'):
        summarize_stress(result, doc)
    doc['cases'].pop(); result['results'].pop()
    doc['cases'][1]['condition'] = CONDITIONS[1]
    with pytest.raises(ValueError, match='every stress condition'):
        summarize_stress(result, doc)


def test_stress_summary_retains_grouping_and_each_condition(tmp_path, monkeypatch):
    from lab.research_benchmark import load_manifest, run
    import lab.research_benchmark as benchmark
    manifest = source(tmp_path)
    doc = build_stress(manifest, tmp_path, tmp_path/'stress')
    monkeypatch.setattr(benchmark, 'optional_webrtc', lambda x: {'status': 'available', 'intervals': []})
    cases = load_manifest(tmp_path/'stress/manifest.json', tmp_path)
    result = run(cases, repetitions=100)
    summary = summarize_stress(result, doc)
    assert result['source_groups'] == 1
    assert all(s['case_count'] == 1 for s in summary['conditions'].values())
    assert all(s['connected_source_groups'] == 1 for s in summary['conditions'].values())
    assert summary['conditions']['original']['analyzed_seconds'] == 1
    assert summary['conditions']['prefix137ms']['analyzed_seconds'] == 1.12


def test_prefix_endpoint_rounding_never_exceeds_decoded_duration():
    from lab.research_metrics import validate_intervals
    x = np.ones(round(8.832*RATE), np.float32)*.1
    y, labels, _ = transform(x, [{'start': 0, 'end': len(x)/RATE}], 'prefix137ms', 0)
    validate_intervals(labels, len(y)/RATE, nonoverlap=True)
    assert labels[-1]['end'] == len(y)/RATE
