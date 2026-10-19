import numpy as np
import pytest
import torch

from lab.antideepfake import AntiDeepfakeDetector, convert, prepare
from lab.challenge import default_fakes
from lab.forensics import abstention_band, fmt_p
from lab.fusion import FusionDetector, design, file_score, fit_logistic, noise_index, sigmoid, window_probabilities


def test_fairseq_keys_map_to_transformers_and_quantizer_is_dropped():
    t = torch.zeros(1)
    state = {'m_ssl.model.feature_extractor.conv_layers.0.0.weight': t, 'm_ssl.model.feature_extractor.conv_layers.0.2.1.bias': t,
             'm_ssl.model.layer_norm.weight': t, 'm_ssl.model.post_extract_proj.bias': t, 'm_ssl.model.mask_emb': t,
             'm_ssl.model.encoder.pos_conv.0.weight_g': t, 'm_ssl.model.encoder.pos_conv.0.weight_v': t,
             'm_ssl.model.encoder.layers.3.self_attn.q_proj.weight': t, 'm_ssl.model.encoder.layers.3.self_attn_layer_norm.bias': t,
             'm_ssl.model.encoder.layers.3.fc1.weight': t, 'm_ssl.model.encoder.layers.3.fc2.bias': t,
             'm_ssl.model.encoder.layers.3.final_layer_norm.weight': t, 'm_ssl.model.encoder.layer_norm.bias': t,
             'm_ssl.model.quantizer.vars': t, 'm_ssl.model.project_q.weight': t, 'm_ssl.model.final_proj.bias': t,
             'proj_fc.weight': t}
    backbone, head = convert(state)
    assert set(backbone) == {
        'feature_extractor.conv_layers.0.conv.weight', 'feature_extractor.conv_layers.0.layer_norm.bias',
        'feature_projection.layer_norm.weight', 'feature_projection.projection.bias', 'masked_spec_embed',
        'encoder.pos_conv_embed.conv.parametrizations.weight.original0', 'encoder.pos_conv_embed.conv.parametrizations.weight.original1',
        'encoder.layers.3.attention.q_proj.weight', 'encoder.layers.3.layer_norm.bias',
        'encoder.layers.3.feed_forward.intermediate_dense.weight', 'encoder.layers.3.feed_forward.output_dense.bias',
        'encoder.layers.3.final_layer_norm.weight', 'encoder.layer_norm.bias'}
    assert set(head) == {'weight'}
    with pytest.raises(ValueError, match='Unexpected'):
        convert({'something.else': t})


def test_antideepfake_preprocessing_and_class_order():
    x = np.random.default_rng(1).normal(0, .1, 5000).astype(np.float32)
    assert np.allclose(prepare(x), torch.nn.functional.layer_norm(torch.from_numpy(x), (len(x),)).numpy(), atol=1e-5)
    with pytest.raises(ValueError):
        prepare(np.zeros(64001, np.float32))
    detector = object.__new__(AntiDeepfakeDetector)
    detector.torch = torch
    detector.logits = lambda x: torch.tensor([2.0, -1.0])  # native order: fake, real
    result = detector.score(x)
    assert result['logits'] == [-1.0, 2.0]  # reported as human, synthetic
    assert result['synthetic_score'] == pytest.approx(float(torch.softmax(torch.tensor([2.0, -1.0]), 0)[0]))


def test_logistic_fit_is_balanced_and_file_score_averages_windows():
    rng = np.random.default_rng(0)
    real = rng.normal([-4, -3], 1, (300, 2))
    fake = rng.normal([4, 3], 1, (30, 2))  # imbalanced on purpose
    X, y = np.vstack([real, fake]), np.r_[np.zeros(300), np.ones(30)]
    w = np.where(y == 1, 1/30, 1/300)
    beta = fit_logistic(X, y, w)
    assert beta[1] > 0 and beta[2] > 0
    p = window_probabilities(beta, X)
    assert p[y == 1].mean() > .9 and p[y == 0].mean() < .1
    assert file_score(beta, X[:2]) == pytest.approx(float(np.mean(p[:2])))
    assert sigmoid(-800) == pytest.approx(0) and sigmoid(800) == pytest.approx(1)


def test_fusion_combines_both_logits():
    fusion = object.__new__(FusionDetector)
    fusion.beta = np.array([-1.0, .5, .25])
    fusion.noise_aware = False

    class Pella:
        def score(self, x):
            return {'synthetic_score': float(sigmoid(4)), 'logits': [0.0, 4.0]}

    class ADF:
        def logits(self, x):
            return torch.tensor([3.0, 1.0])  # fake - real = 2
    fusion.pella, fusion.adf = Pella(), ADF()
    result = fusion.score(np.zeros(100, np.float32))
    assert result['logits'] == [0.0, pytest.approx(-1 + .5*4 + .25*2)]
    assert result['synthetic_score'] == pytest.approx(float(sigmoid(1.5)))
    assert result['components'] == {'pellav2': pytest.approx(float(sigmoid(4))), 'antideepfake': pytest.approx(float(sigmoid(2)))}


def test_logit_band_stays_inside_unit_interval_near_one():
    low, high = abstention_band(.9992, 2.0)
    centre = np.log(.9992/.0008)
    assert 0 < low < .9992 < high < 1
    assert np.log(low/(1-low)) == pytest.approx(centre - 2) and np.log(high/(1-high)) == pytest.approx(centre + 2)
    assert abstention_band(.5) == (.35, .65)
    assert fmt_p(.5) == '0.50' and fmt_p(.99917) == '0.9992'


def test_v4_challenge_weights_the_heldout_generator():
    manifest = {'heldout_generator': 'xtts', 'cases': [{'label': 'synthetic', 'generator': g} for g in ('chatterbox', 'f5', 'xtts')]}
    assert default_fakes(manifest) == {'chatterbox': 3, 'f5': 3, 'xtts': 4}
    assert default_fakes({'cases': [{'label': 'synthetic', 'generator': g} for g in ('chatterbox', 'f5')]}) == {'chatterbox': 5, 'f5': 5}


def test_fusion_bar_uses_heldout_limit():
    from scripts.fit_fusion import bar_failures
    cell = lambda n, k: {'flagged_synthetic': k, 'n': n, 'rate': k/n}
    report = {'clean': {'by_generator': {'f5': cell(30, 29), 'xtts': cell(60, 55)}, 'reals': cell(200, 4)}}
    assert bar_failures(report, heldout='xtts') == []
    assert bar_failures(report) == ['clean/xtts: 55/60 fakes caught < 95%']
    report['clean']['reals'] = cell(200, 5)
    assert 'clean: 5/200 real flagged > 2%' in bar_failures(report, heldout='xtts')


def test_noise_index_and_noise_aware_design():
    rng = np.random.default_rng(0)
    t = np.arange(64000) / 16000
    speech = (np.sin(2*np.pi*200*t) * (np.sin(2*np.pi*.7*t) > 0)).astype(np.float32) * .3  # tone bursts with silent pauses
    assert noise_index(speech) == 0.0
    noisy = speech + rng.normal(0, np.sqrt(np.mean(speech**2))*10**(-20/20), len(speech)).astype(np.float32)  # 20 dB SNR
    assert noise_index(noisy) > .5
    assert noise_index(rng.normal(0, .1, 64000)) > 1.9  # pure noise: almost no level spread
    assert design([2.0, -1.0, .5]).tolist() == [[1, 2, -1, .5, 1.0, -.5]]
    assert design([[40.0, -1.0]]).tolist() == [[1, 30, -1]]  # logits clipped, base design unchanged
