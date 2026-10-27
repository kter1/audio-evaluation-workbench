import pytest

from lab.runtime_metrics import peak_rss_gib, timing_summary


def test_runtime_weighted_by_audio_and_stratified():
    result = timing_summary([{'audio_seconds': 1., 'wall_seconds': .1},
                             {'audio_seconds': 10., 'wall_seconds': .2}])
    assert result['realtime_factor'] == pytest.approx(.3/11)
    assert result['latency_p50_seconds'] == pytest.approx(.15)
    assert result['duration_bins']['<5s']['calls'] == 1
    assert result['duration_bins']['5-15s']['calls'] == 1


@pytest.mark.parametrize('rows', [[], [{'audio_seconds': 0, 'wall_seconds': 1}],
                                 [{'audio_seconds': 1, 'wall_seconds': float('nan')}]])
def test_runtime_rejects_undefined_measurements(rows):
    with pytest.raises(ValueError): timing_summary(rows)


def test_peak_rss_platform_units():
    assert peak_rss_gib(1024**3, 'Darwin') == 1
    assert peak_rss_gib(1024**2, 'Linux') == 1
    with pytest.raises(ValueError): peak_rss_gib(100, 'unknown')
