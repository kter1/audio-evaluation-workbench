import numpy as np
import pytest
from scipy.signal import correlate, correlation_lags

from lab.noise_candidates import preprocess


@pytest.mark.parametrize('candidate',['spectral_floor','lowpass_control'])
def test_candidates_preserve_timing_and_finite_output(candidate):
    x=np.random.default_rng(19).normal(0,.1,16001).astype('float32')
    y=preprocess(x,candidate)
    assert len(y)==len(x) and y.dtype==np.float32 and np.isfinite(y).all()
    assert np.array_equal(y,preprocess(x,candidate))
    lag=correlation_lags(len(y),len(x))[np.argmax(correlate(y,x,method='fft'))]
    assert lag==0
    assert not np.count_nonzero(preprocess(np.zeros_like(x),candidate))


def test_lowpass_control_has_expected_analytic_frequency_response():
    t=np.arange(32000)/16000
    low=np.sin(2*np.pi*500*t).astype('float32')*.1
    high=np.sin(2*np.pi*6000*t).astype('float32')*.1
    low_out=preprocess(low,'lowpass_control')[100:-100]
    high_out=preprocess(high,'lowpass_control')[100:-100]
    assert np.sqrt(np.mean(low_out**2))==pytest.approx(.1/np.sqrt(2),rel=.01)
    assert np.sqrt(np.mean(high_out**2))<.001


@pytest.mark.parametrize('bad',[np.zeros(7999), np.ones((16000,2)), np.full(16000,np.nan)])
def test_candidates_refuse_invalid_analysis_input(bad):
    with pytest.raises(ValueError):preprocess(bad,'spectral_floor')


def test_neural_candidate_refuses_unverified_checkpoint_before_loading(tmp_path):
    from lab.research_metricgan import MetricGANSession
    (tmp_path/'enhance_model.ckpt').write_bytes(b'not a verified checkpoint')
    with pytest.raises(ValueError,match='size mismatch'):
        MetricGANSession(tmp_path)
