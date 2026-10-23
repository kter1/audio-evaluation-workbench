import ast
from hashlib import sha1
import json

import numpy as np
import pytest

from lab.research_rawnetlite import ROOT, REVISION, RawNetLiteDetector, preprocess


@pytest.mark.parametrize('length', [8000, 48000, 64000])
def test_preprocessing_matches_pinned_author_function_at_16k(length):
    torch = pytest.importorskip('torch')
    path = ROOT/'data/research-tools/rawnetlite'/REVISION/'audio_preprocessor.py'
    if not path.exists():
        pytest.skip('Optional pinned research source is not acquired')
    raw = path.read_bytes()
    spec = json.loads((ROOT/'docs/rawnetlite-acquisition-v1.json').read_text())
    entry = next(f for f in spec['files'] if f['name'] == path.name)
    assert sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == entry['git_blob_sha1']
    # Execute only the inspected pure preprocessing function, without file IO,
    # argument parsing, downloads, TorchAudio imports or corpus operations.
    tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_audio')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    x = np.random.default_rng(412).normal(0, .1, length).astype('float32')
    # Exercise normalization before trimming: the largest value is in the tail.
    x[-1] = 2
    reference = namespace['preprocess_audio'](torch.from_numpy(x)[None], 16000)
    actual = preprocess(x)
    assert actual.shape == (1, 48000)
    assert torch.equal(actual, reference)


def test_zero_audio_has_defined_finite_preprocessing():
    pytest.importorskip('torch')
    result = preprocess(np.zeros(8000, dtype='float32')).numpy()
    assert result.shape == (1, 48000) and not np.any(result)


@pytest.mark.parametrize('samples', [[], [[.1]], [float('nan')], [float('inf')]])
def test_invalid_audio_rejected_before_torch(samples):
    with pytest.raises(ValueError, match='finite mono16k'):
        preprocess(samples)


def test_corrupt_checkpoint_rejected_before_deserialization(tmp_path):
    directory = tmp_path/'models'
    directory.mkdir()
    (directory/'augmented_triple_cross_domain_focal_rawnet_lite.pt').write_bytes(b'bad')
    with pytest.raises(ValueError, match='file size mismatch'):
        RawNetLiteDetector(tmp_path)


def test_same_size_checkpoint_tampering_is_rejected(tmp_path):
    directory = tmp_path/'models'
    directory.mkdir()
    (directory/'augmented_triple_cross_domain_focal_rawnet_lite.pt').write_bytes(bytes(987460))
    with pytest.raises(ValueError, match='Git blob identity'):
        RawNetLiteDetector(tmp_path)
