from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf

from lab.forensics import RATE
from lab.store import digest
from scripts.screen_partial_synthesis import prepare_challenge, screen, selection


def fixture_manifest(tmp_path):
    cases = []
    for speaker in range(4):
        for kind, seconds, frequency in (('context', 14, 110), ('fake', 5, 700), ('control', 5, 230)):
            x = (.08*np.sin(np.arange(seconds*RATE)*2*np.pi*(frequency+3*speaker)/RATE)).astype('float32')
            path = tmp_path/f'{speaker}-{kind}.wav'
            sf.write(path, x, RATE, subtype='FLOAT')
            cases.append({'id': path.stem, 'path': path.name, 'sha256': sha256(path.read_bytes()).hexdigest(),
                'duration': seconds, 'speaker': str(speaker),
                'source_utt': f'{speaker}-control' if kind == 'control' else f'{speaker}-matched',
                'ref_utt': f'{speaker}-reference' if kind == 'fake' else None,
                'label': 'synthetic' if kind == 'fake' else 'human',
                'generator': 'fixture-tts' if kind == 'fake' else 'bonafide'})
    document = {'version': '4.0', 'cases': cases}
    document['sha256'] = digest(document)
    return document


def test_selection_is_order_independent_and_uses_distinct_contexts(tmp_path):
    doc = fixture_manifest(tmp_path)
    expected = [pair[0] for pair in selection(doc)]
    doc['cases'].reverse()
    doc['sha256'] = digest({k: v for k, v in doc.items() if k != 'sha256'})
    assert [pair[0] for pair in selection(doc)] == expected
    assert len(set(pair[1]['id'] for pair in selection(doc))) == 4


def test_splices_preserve_context_and_use_matched_genuine_boundaries(tmp_path):
    inputs, identities, pairs = prepare_challenge(fixture_manifest(tmp_path), tmp_path)
    assert len(inputs) == 28 and len(identities) == 28 and len(pairs) == 4
    by_pair = {}
    for samples, metadata in inputs:
        by_pair.setdefault(metadata['pair_id'], {})[metadata['condition']] = (samples, metadata)
        assert len(samples) == 12*RATE
        assert sha256(samples.astype('<f4').tobytes()).hexdigest() == metadata['analysis_samples_sha256']
    for variants in by_pair.values():
        original = variants['original'][0]
        for seconds in (1, 2, 4):
            fake, meta = variants[f'partial_{seconds}s']
            genuine, control = variants[f'genuine_splice_{seconds}s']
            start, end = meta['splice']['start_sample_16k'], meta['splice']['end_sample_16k']
            assert end-start == seconds*RATE
            assert (start, end) == (control['splice']['start_sample_16k'], control['splice']['end_sample_16k'])
            assert np.array_equal(fake[:start], original[:start]) and np.array_equal(fake[end:], original[end:])
            assert np.array_equal(genuine[:start], original[:start]) and np.array_equal(genuine[end:], original[end:])
            assert not np.array_equal(fake[start:end], genuine[start:end])
    assert all(any(t.startswith('librispeech:utterance:') for t in r['identity_tokens']) for r in identities)


def test_source_tampering_fails_before_model_inference(tmp_path):
    doc = fixture_manifest(tmp_path)
    first = selection(doc)[0][2]
    (tmp_path/first['path']).write_bytes(b'changed')
    class Detector:
        threshold = .5
        manifest = {}
        def score(self, samples):
            pytest.fail('Inference must not run with modified sources')
    with pytest.raises(ValueError, match='source hash mismatch'):
        screen(doc, tmp_path, Detector())


def test_full_context_windows_and_failure_counts_are_distinct(tmp_path):
    class Detector:
        threshold = .5
        window_samples = 4*RATE
        manifest = {'fixture': True}
        calls = 0
        def score(self, samples):
            assert len(samples) == 4*RATE
            self.calls += 1
            # A constant genuine vote exposes every partial's false reassurance.
            return {'synthetic_score': .001}
    detector = Detector()
    result = screen(fixture_manifest(tmp_path), tmp_path, detector)
    assert detector.calls == 28*3
    assert result['source_groups'] == 4
    assert result['outcomes']['false_human_indications_on_partial'] == 12
    assert result['outcomes']['false_synthetic_indications_on_genuine_controls'] == 0
    assert result['outcomes']['genuine_controls'] == 16
    assert all(len(row['windows']) == 3 for row in result['results'])


def test_insufficient_long_contexts_rejected_without_repetition(tmp_path):
    doc = fixture_manifest(tmp_path)
    doc['cases'][0]['duration'] = 11
    doc['sha256'] = digest({k: v for k, v in doc.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='Need four distinct'):
        selection(doc)
