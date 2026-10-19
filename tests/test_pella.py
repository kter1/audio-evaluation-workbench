import numpy as np
import pytest
import torch

from lab.evaluation import detection_rate, summarize, wer
from lab.forensics import RATE, anchored_spans, decision, diagnostics
from lab.pella import WINDOW, PellaV2Detector, prepare
from lab.vendor.pellav2_model import Detector


def test_padding_happens_before_standardization():
    x = np.array([1, 2, 3], dtype=np.float32)
    padded = np.array([1, 2, 3, 0, 0, 0, 0, 0], dtype=np.float32)
    expected = (padded-padded.mean())/(padded.std()+1e-7)
    assert np.allclose(prepare(x, 8), expected)
    full = np.linspace(-1, 1, 8, dtype=np.float32)
    assert np.allclose(prepare(full, 8), (full-full.mean())/(full.std()+1e-7))
    for bad in [np.array([], np.float32), np.zeros(9, np.float32), np.array([np.nan], np.float32)]:
        with pytest.raises(ValueError):
            prepare(bad, 8)


def test_single_logit_reports_as_two_class_softmax():
    detector = object.__new__(PellaV2Detector)
    detector.torch = torch
    for logit in [-30., -2.5, 0., 1.25, 30.]:
        detector.model = lambda x, l=logit: torch.tensor([l])
        result = detector.score(np.ones(100, np.float32))
        assert result['logits'] == [0.0, logit]
        assert result['synthetic_score'] == pytest.approx(1/(1+np.exp(-logit)))
        assert result['synthetic_score'] == pytest.approx(float(torch.softmax(torch.tensor(result['logits']), 0)[1]))
        assert result['human_score'] == pytest.approx(1-result['synthetic_score'])


@pytest.mark.parametrize('n', [1, 63999, 64000, 64001, 200000])
def test_anchored_spans_cover_audio_with_full_context(n):
    spans = anchored_spans(n, WINDOW)
    assert spans[0][0] == 0 and spans[-1][1] == n
    assert all(b-a == min(n, WINDOW) for a, b in spans)
    assert len({a for a, _ in spans}) == len(spans)
    assert all(spans[i][1] >= spans[i+1][0] for i in range(len(spans)-1))


def test_checksums_are_enforced(tmp_path):
    with pytest.raises(ValueError, match='not installed'):
        PellaV2Detector(tmp_path)
    for name in ['pellav2_detector.pt', 'pellav2_infer.py', 'config.json', 'xls-r-300m-config.json']:
        (tmp_path/name).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        PellaV2Detector(tmp_path)


def test_vendored_architecture_runs_on_a_tiny_backbone():
    from transformers import Wav2Vec2Config
    config = Wav2Vec2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                            conv_dim=(8, 8), conv_stride=(5, 2), conv_kernel=(10, 3), feat_extract_norm='layer',
                            do_stable_layer_norm=True, num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2)
    model = Detector(config.to_dict()).eval()
    assert model.layer_weights.shape == (3,)
    with torch.inference_mode():
        out = model(torch.from_numpy(prepare(np.random.default_rng(0).normal(size=4000).astype(np.float32), 4000))[None])
    assert out.shape == (1,) and torch.isfinite(out).all()


def test_decision_band_follows_the_detector_threshold():
    metrics = diagnostics((.2*np.sin(np.arange(3*RATE)/7)).astype(np.float32)[:, None], RATE)[1]
    at = lambda score, threshold: decision([{'start': 0, 'end': 3, 'synthetic_score': score}], metrics, 3, threshold)
    assert at(.36, .5)['assessment'] == 'Inconclusive'
    assert at(.34, .5)['assessment'] == 'Human speech indicated'
    shifted = at(.7, .6)
    assert shifted['binary_model_decision'] == 'synthetic' and shifted['assessment'] == 'Inconclusive'
    assert at(.76, .6)['assessment'] == 'Synthetic speech indicated'
    assert at(.55, .6)['binary_model_decision'] == 'human'


def test_threshold_aware_summaries_and_detection_rate():
    rows = [{'label': 'human', 'score': .55, 'assessment': 'x'}, {'label': 'synthetic', 'score': .58, 'assessment': 'x'}]
    assert summarize(rows)['correct'] == 1
    assert summarize(rows, threshold=.56)['correct'] == 2
    cell = detection_rate([{'score': .9}, {'score': .2}, {'score': .5}])
    assert cell['flagged_synthetic'] == 2 and cell['rate'] == pytest.approx(2/3) and cell['n'] == 3
    assert detection_rate([])['rate'] is None


def test_word_error_rate():
    assert wer('The cat sat.', 'the cat sat') == 0
    assert wer('the cat sat', 'the cat') == pytest.approx(1/3)
    assert wer('the cat sat', 'a cat sat on') == pytest.approx(2/3)
    assert wer("DON'T STOP", "don't stop") == 0
    with pytest.raises(ValueError):
        wer('...', 'anything')
