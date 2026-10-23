import ast
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

from lab.research_pella_current import (
    ROOT, REVISION, SR, WINDOW, FILES, PellaCurrentDetector,
    author_detector_type, load_kaldi, preprocess, verify_file, validate_state,
)

DIRECTORY = ROOT/'data/research-tools/pella-current'/REVISION


def source_directory():
    if not (DIRECTORY/'pellav2_infer.py').exists():
        pytest.skip('Optional pinned current Pellav2 source is not acquired')
    return DIRECTORY


@pytest.mark.parametrize('samples', [[], [[.1]], [float('nan')], [float('inf')]])
def test_invalid_audio_is_rejected(samples):
    with pytest.raises(ValueError, match='finite mono16k'):
        preprocess(samples)


def test_corrupt_checkpoint_rejected_before_source_execution(tmp_path):
    (tmp_path/'pellav2_detector.pt').write_bytes(b'bad')
    with pytest.raises(ValueError, match='file size mismatch'):
        PellaCurrentDetector(tmp_path)


def test_same_size_source_tampering_rejected_before_execution(tmp_path):
    (tmp_path/'pellav2_infer.py').write_bytes(b' '*FILES['pellav2_infer.py'][0])
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        verify_file(tmp_path, 'pellav2_infer.py')


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_saved_weights_are_rejected_before_model_construction(bad):
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='nonfinite weights: backbone.weight'):
        validate_state({'backbone.weight': torch.tensor([1., bad])})


def test_non_tensor_checkpoint_values_are_rejected():
    pytest.importorskip('torch')
    with pytest.raises(ValueError, match='non-tensor: unexpected'):
        validate_state({'unexpected': 'not a tensor'})


@pytest.mark.parametrize('length', [480, 8000, 64000, 70001])
def test_mono16k_preprocessing_matches_pinned_author(length, monkeypatch):
    directory = source_directory()
    verify_file(directory, 'pellav2_infer.py')
    tree = ast.parse((directory/'pellav2_infer.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('loudness', 'load_clip')]
    x = np.random.default_rng(514).normal(.08, .14, length).astype('float32')
    x[-1] = 1.2
    # Already-mono16k fixture: source's resampling branch is not exercised.
    monkeypatch.setitem(sys.modules, 'torchaudio', ModuleType('torchaudio'))
    namespace = {'np': np, 'sf': SimpleNamespace(read=lambda *a, **k: (x.copy(), SR)),
                 'SR': SR, 'CROP': WINDOW, 'TARGET_LUFS': -23.}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'pinned-load-clip', 'exec'), namespace)
    expected = namespace['load_clip']('unused.wav')
    actual = preprocess(x)
    assert actual.shape == (WINDOW,) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_zero_audio_stays_zero_and_finite():
    actual = preprocess(np.zeros(8000, dtype='float32'))
    assert actual.shape == (WINDOW,) and not np.any(actual)


def test_official_fbank_is_deterministic_and_uses_80_bins():
    torch = pytest.importorskip('torch')
    directory = source_directory()
    if not (directory/'kaldi.py').exists():
        pytest.skip('Optional pinned official Kaldi source is not acquired')
    kaldi = load_kaldi(directory)
    x = torch.from_numpy(np.random.default_rng(907).normal(0, .1, WINDOW).astype('float32'))[None]
    first = kaldi.fbank(x*32768, num_mel_bins=80, sample_frequency=SR,
                        dither=0., window_type='povey')
    second = kaldi.fbank(x*32768, num_mel_bins=80, sample_frequency=SR,
                         dither=0., window_type='povey')
    assert first.shape == (398, 80)
    assert torch.equal(first, second) and torch.isfinite(first).all()


def test_author_features_keep_sample_variance_even_frames_and_padding_mask():
    torch = pytest.importorskip('torch')
    directory = source_directory()
    # A known fbank fixture distinguishes sample variance from population variance,
    # adjacent-frame concatenation from stacking, and true length from zero padding.
    calls = []
    values = torch.arange(7*80, dtype=torch.float32).reshape(7, 80)
    def fbank(waveform, **kwargs):
        calls.append((waveform.clone(), kwargs))
        return values.clone() if len(calls) == 1 else values[:5].clone()
    Author = author_detector_type(directory, SimpleNamespace(fbank=fbank))
    samples = torch.zeros(2, WINDOW)
    samples[0, :8000] = .2
    samples[1, :4000] = -.1
    features, mask = Author.features(samples)
    first = (values-values.mean(0, keepdim=True))/torch.sqrt(values.var(0, keepdim=True)+1e-7)
    second = values[:5]
    second = (second-second.mean(0, keepdim=True))/torch.sqrt(second.var(0, keepdim=True)+1e-7)
    assert features.shape == (2, 3, 160)
    assert torch.equal(features[0], first[:6].reshape(3, 160))
    assert torch.equal(features[1, :2], second[:4].reshape(2, 160))
    assert not features[1, 2].any()
    assert torch.equal(mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert calls[0][0].shape == (1, 8000)
    assert calls[1][0].shape == (1, 4000)
    assert calls[0][1] == {'num_mel_bins': 80, 'sample_frequency': SR,
                           'dither': 0., 'window_type': 'povey'}


def test_author_forward_masks_padding_and_uses_learned_layer_weights():
    torch = pytest.importorskip('torch')
    directory = source_directory()
    Author = author_detector_type(directory, SimpleNamespace())
    model = Author()
    model.layer_weights = torch.nn.Parameter(torch.log(torch.tensor([1., 3.])))
    model.head = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.head.weight[:] = torch.tensor([[1., 2.]])
    first = torch.tensor([[[1., 2.], [100., 100.]]])
    second = torch.tensor([[[3., 4.], [200., 200.]]])
    model.features = lambda wav: (torch.zeros(1, 2, 160), torch.tensor([[1, 0]]))
    model.backbone = lambda **kwargs: SimpleNamespace(hidden_states=(first, second))
    actual = model(torch.zeros(1, WINDOW))
    # Weighted unpadded state = .25*[1,2]+.75*[3,4] = [2.5,3.5].
    torch.testing.assert_close(actual, torch.tensor([9.5]), rtol=1e-6, atol=1e-6)
