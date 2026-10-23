import ast
from types import SimpleNamespace

import numpy as np
import pytest

from lab.research_wavlm_aasist import (
    ROOT, REVISION, WINDOW, SOURCE_FILES, WavLMAASISTDetector,
    author_classes, preprocess, validate_state, verify_source, inference_state, UNUSED_PROJECTION,
)

DIRECTORY = ROOT/'data/research-tools/wavlm-aasist'/REVISION


def source_directory():
    if not (DIRECTORY/'inference.py').exists():
        pytest.skip('Optional pinned WavLM/AASIST research source is not acquired')
    return DIRECTORY


@pytest.mark.parametrize('samples', [[], [[.1]], [float('nan')], [float('inf')]])
def test_invalid_audio_rejected(samples):
    with pytest.raises(ValueError, match='finite mono16k'):
        preprocess(samples)


@pytest.mark.parametrize('length', [8000, 64000, 80000, 90001])
def test_preprocessing_matches_pinned_author_mono16k_path(length):
    torch = pytest.importorskip('torch')
    directory = source_directory()
    verify_source(directory, 'inference.py')
    tree = ast.parse((directory/'inference.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_audio')
    x = np.random.default_rng(641).normal(0, .1, length).astype('float32')
    x[-1] = 2.  # Ensures normalization precedes cropping.
    namespace = {'torch': torch, 'torchaudio': SimpleNamespace(
        load=lambda path: (torch.from_numpy(x.copy())[None], 16000))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'pinned-load-audio', 'exec'), namespace)
    expected = namespace['load_audio']('unused.wav')
    actual = preprocess(x)
    assert actual.shape == (1, WINDOW)
    assert torch.equal(actual[0], expected)


def test_zero_audio_stays_zero():
    pytest.importorskip('torch')
    actual = preprocess(np.zeros(1000, dtype='float32'))
    assert actual.shape == (1, WINDOW) and not actual.any()


def test_corrupt_checkpoint_rejected_before_loading(tmp_path):
    (tmp_path/'checkpoint_epoch_5.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError, match='checkpoint size mismatch'):
        WavLMAASISTDetector(tmp_path)


def test_same_size_source_tampering_rejected(tmp_path):
    (tmp_path/'model.py').write_bytes(b' '*SOURCE_FILES['model.py'][0])
    with pytest.raises(ValueError, match='Git blob identity'):
        verify_source(tmp_path, 'model.py')


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_saved_weights_rejected(bad):
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='nonfinite weights: head.weight'):
        validate_state({'head.weight': torch.tensor([1., bad])})


def test_unexpected_checkpoint_key_is_not_silently_ignored():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='Unexpected WavLM/AASIST checkpoint keys: unreviewed.weight'):
        inference_state({'head.weight': torch.ones(1), 'unreviewed.weight': torch.ones(1)}, {'head.weight'})


def test_unused_projection_names_alone_do_not_bypass_tensor_authentication():
    torch = pytest.importorskip('torch')
    state = {name: torch.zeros(shape) for name, (shape, digest) in UNUSED_PROJECTION.items()}
    with pytest.raises(ValueError, match='projection SHA-256 mismatch'):
        inference_state(state, set())


@pytest.mark.parametrize('native', [-4., 4.])
def test_score_uses_real_native_label_direction(native):
    torch = pytest.importorskip('torch')
    detector = object.__new__(WavLMAASISTDetector)
    detector.torch = torch
    detector.model = lambda wav: torch.tensor([native])
    detector.timings = []
    score = detector.score(np.ones(8000, dtype='float32'))
    assert score['synthetic_score'] == 1.-float(torch.sigmoid(torch.tensor(native)))
    assert score['logits'] == [0., -native]
    assert (score['synthetic_score'] >= .5) == (native <= 0)


def test_forward_excludes_embedding_and_preserves_layer_order():
    torch = pytest.importorskip('torch')
    classes = author_classes(source_directory())
    model = classes['DeepfakeDetector']()
    hidden = tuple(torch.full((1, 2, 3), float(i)) for i in range(4))
    model.wavlm = lambda waveform, **kwargs: SimpleNamespace(hidden_states=hidden)
    captured = []
    def pool(states):
        captured.extend(states)
        return torch.ones(1, 5)
    model.pool = pool
    model.classifier = lambda pooled: torch.tensor([[2.]])
    actual = model(torch.ones(1, WINDOW))
    assert torch.equal(actual, torch.tensor([2.]))
    assert len(captured) == 3
    assert all(torch.equal(value, hidden[i+1]) for i, value in enumerate(captured))
