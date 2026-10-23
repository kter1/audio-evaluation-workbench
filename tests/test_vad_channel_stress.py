from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pytest
import soundfile as sf

from lab.forensics import diagnostics, mono16
from lab.vad_channel_stress import crop_labels, CHANNEL_CONDITIONS
from lab.vad_stress import build_stress, transform


def test_stereo_conversion_and_cancellation_warning():
    x = np.sin(np.arange(16000)*.1).astype(np.float32)*.2
    empty, _, _ = transform(x, [], 'stereo_one_empty', 1)
    assert np.array_equal(mono16(empty, 16000), x*.5)
    opposite, _, _ = transform(x, [], 'stereo_antiphase', 1)
    assert np.count_nonzero(mono16(opposite, 16000)) == 0
    _, metrics, warnings = diagnostics(opposite, 16000)
    assert any('Opposite-phase' in warning for warning in warnings)
    assert metrics['stereo_correlation'] == pytest.approx(-1)


def test_crop_labels_intersect_shift_and_preserve_unknowns():
    intervals = [{'start': .1, 'end': .3}, {'start': .4, 'end': .9}, {'start': 1.1, 'end': 1.5}]
    assert crop_labels(intervals, 8000, 16000) == [{'start': 0., 'end': .4}]
    assert crop_labels(None, 0, 8000) is None
    assert crop_labels([], 0, 8000) == []
    x = np.arange(16000, dtype=np.float32)/16000
    y, labels, metadata = transform(x, intervals[:2], 'crop_center500ms', 1)
    assert np.array_equal(y, x[4000:12000])
    assert metadata['crop_start_sample'] == 4000
    assert labels == [{'start': 0., 'end': pytest.approx(.05)}, {'start': pytest.approx(.15), 'end': .5}]


def test_bandwidth_roundtrip_suppresses_high_frequency_and_preserves_count():
    time = np.arange(16001)/16000
    low = np.sin(2*np.pi*1000*time).astype('float32')*.1
    high = np.sin(2*np.pi*6000*time).astype('float32')*.1
    ylow, _, _ = transform(low, [], 'bandwidth8k', 0)
    yhigh, _, _ = transform(high, [], 'bandwidth8k', 0)
    assert len(ylow) == len(yhigh) == len(low)
    assert np.sqrt(np.mean(yhigh[100:-100]**2)) < .001*np.sqrt(np.mean(ylow[100:-100]**2))


def test_codec_roundtrip_checks_length_alignment_and_retains_encoded_payload():
    x = np.random.default_rng(17).normal(0,.08,16000).astype('float32')
    y, labels, metadata = transform(x, [], 'vorbis_roundtrip', 0)
    assert len(y) == len(x) and labels == []
    assert not np.array_equal(y, x)
    assert metadata['decoded_global_correlation_lag_samples'] == 0
    assert sha256(metadata['codec_payload']).hexdigest() == metadata['encoded_sha256']
    decoded, rate = sf.read(BytesIO(metadata['codec_payload']), dtype='float32')
    assert np.array_equal(y, decoded) and rate == 16000


def test_extended_preparation_keeps_grouping_and_codec_asset(tmp_path, monkeypatch):
    x = np.random.default_rng(9).normal(0,.08,32000).astype('float32')
    sf.write(tmp_path/'source.wav', x, 16000, subtype='FLOAT')
    manifest = tmp_path/'source.json'
    manifest.write_text(json.dumps({'schema_version': 'method-benchmark-1.0', 'partition': 'development',
        'cases': [{'id': 'source', 'path': 'source.wav', 'sha256': sha256((tmp_path/'source.wav').read_bytes()).hexdigest(),
                   'source_group': 'group', 'annotation_provenance': 'fixture', 'speech_intervals': [],
                   'level_events': None, 'noise_reference_dbfs': None}]}))
    result = build_stress(manifest, tmp_path, tmp_path/'stress', extended=True)
    assert len(result['cases']) == len(CHANNEL_CONDITIONS)
    assert all(row['source_group'] == 'group' for row in result['cases'])
    codec = next(row for row in result['cases'] if row['condition'] == 'vorbis_roundtrip')['transformation']
    assert sha256((tmp_path/codec['encoded_path']).read_bytes()).hexdigest() == codec['encoded_sha256']
    assert 'codec_payload' not in codec
    from lab.research_benchmark import load_manifest
    cases = load_manifest(tmp_path/'stress/manifest.json', tmp_path)
    assert len(cases) == 8
    canceled = next(c for c in cases if c['condition'] == 'stereo_antiphase')
    assert np.count_nonzero(canceled['samples']) == 0
    from lab.research_benchmark import run
    from lab.vad_stress import summarize_stress
    import lab.research_benchmark as benchmark
    monkeypatch.setattr(benchmark, 'optional_webrtc', lambda x: {'status': 'available', 'intervals': []})
    comparison = run(cases, repetitions=100)
    summary = summarize_stress(comparison, result)
    assert set(summary['conditions']) == set(CHANNEL_CONDITIONS)
    assert comparison['source_groups'] == 1
    from lab.vad_transport_audit import audit_transport
    audit = audit_transport(tmp_path/'stress/manifest.json', tmp_path)
    assert audit['case_count'] == audit['passed_cases'] == 4
    import lab.vad_transport_audit as transport
    original_diagnostics = transport.diagnostics
    def hidden_warnings(*args):
        x, metrics, _ = original_diagnostics(*args)
        return x, metrics, []
    monkeypatch.setattr(transport, 'diagnostics', hidden_warnings)
    assert transport.audit_transport(tmp_path/'stress/manifest.json', tmp_path)['passed_cases'] == 0


def test_codec_alignment_failure_is_not_silently_repaired(monkeypatch):
    import lab.vad_channel_stress as channel
    original_read = sf.read
    def shifted(*args, **kwargs):
        y, rate = original_read(*args, **kwargs)
        return np.r_[np.zeros(10, np.float32), y[:-10]], rate
    monkeypatch.setattr(channel.sf, 'read', shifted)
    x = np.random.default_rng(7).normal(0,.1,16000).astype('float32')
    with pytest.raises(ValueError, match='not zero lag'):
        transform(x, [], 'vorbis_roundtrip', 0)
