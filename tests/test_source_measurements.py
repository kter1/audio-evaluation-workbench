import json

import numpy as np
import pytest

from lab.forensics import diagnostics, mono16
from lab.source_measurements import measure_source


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000, 192000])
def test_native_sine_levels_within_contract_tolerance(rate):
    amplitude = .2
    x = amplitude * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)
    result = measure_source(x[:, None].astype(np.float32), rate)
    channel = result["channels"][0]
    assert abs(channel["rms_dbfs"] - 20 * np.log10(amplitude / np.sqrt(2))) <= .1
    assert abs(channel["sample_peak_dbfs"] - 20 * np.log10(amplitude)) <= .1
    assert abs(channel["crest_factor_db"] - 20 * np.log10(np.sqrt(2))) <= .1
    assert result["sample_rate_hz"] == rate
    assert result["duration_seconds"] == 1
    assert result["arithmetic_mixdown"]["energy_ratio"] == pytest.approx(1)


def test_native_channels_survive_complete_cancellation_and_excess_amplitude():
    x = np.tile([2., -2.], 16000).astype(np.float32)
    stereo = np.stack([x, -x], axis=1)
    mono, metrics, warnings = diagnostics(stereo, 16000)
    assert np.array_equal(mono, mono16(stereo, 16000))
    assert not mono.any()
    native = metrics["source_measurements"]
    assert all(c["rms_amplitude"] == 2 for c in native["channels"])
    assert all(c["over_full_scale_fraction"] == 1 for c in native["channels"])
    assert native["arithmetic_mixdown"]["complete_cancellation"]
    assert native["arithmetic_mixdown"]["energy_ratio"] == 0
    assert native["arithmetic_mixdown"]["attenuation_db"] is None
    assert any("mono mixing masks" in w for w in warnings)
    json.dumps(metrics, allow_nan=False)


def test_source_high_frequency_energy_not_lost_to_analysis_resampling():
    rate = 48000
    x = (.5 * np.sin(2 * np.pi * 12000 * np.arange(rate) / rate)).astype(np.float32)[:, None]
    mono, metrics, _ = diagnostics(x, rate)
    assert np.array_equal(mono, mono16(x, rate))
    assert metrics["source_measurements"]["channels"][0]["rms_amplitude"] == pytest.approx(.5 / np.sqrt(2), abs=1e-6)
    assert np.sqrt(np.mean(mono[100:-100]**2)) < .001
    assert metrics["analysis_representation"]["sample_rate_hz"] == 16000


def test_dc_offset_counts_and_silent_db_values_are_explicit():
    x = np.array([[0, 1.1], [0, -.999], [0, .25], [0, -.25]], dtype=np.float64)
    result = measure_source(x, 8000)
    silent, active = result["channels"]
    assert silent["rms_amplitude"] == 0
    assert silent["rms_dbfs"] is None and silent["sample_peak_dbfs"] is None
    assert silent["crest_factor_db"] is None
    assert active["near_full_scale_samples"] == 2
    assert active["over_full_scale_samples"] == 1
    assert active["near_full_scale_fraction"] == .5
    assert active["dc_offset"] == pytest.approx(x[:, 1].mean())
    json.dumps(result, allow_nan=False)
    silent_mix = measure_source(np.zeros((100, 2)), 8000)["arithmetic_mixdown"]
    assert silent_mix["energy_ratio"] is None
    assert not silent_mix["complete_cancellation"]


def test_chunked_measurements_match_direct_calculation_without_mutating_input():
    x = np.random.default_rng(42).normal(0, .1, (150001, 3)).astype(np.float32)
    original = x.copy()
    result = measure_source(x, 44100)
    for i, channel in enumerate(result["channels"]):
        assert channel["rms_amplitude"] == pytest.approx(np.sqrt(np.mean(x[:, i].astype(float)**2)), abs=1e-12)
        assert channel["dc_offset"] == pytest.approx(x[:, i].astype(float).mean(), abs=1e-12)
    reference = np.mean(x.astype(float)**2)
    mixed = np.mean(x.astype(float).mean(axis=1)**2)
    assert result["arithmetic_mixdown"]["energy_ratio"] == pytest.approx(mixed / reference, abs=1e-12)
    assert np.array_equal(x, original)


@pytest.mark.parametrize("audio,rate", [(np.zeros(100), 16000), (np.zeros((0, 1)), 16000),
    (np.zeros((10, 0)), 16000), (np.zeros((10, 9)), 16000), (np.zeros((10, 1)), 0),
    (np.zeros((10, 1)), 16000.5), (np.full((10, 1), np.nan), 16000)])
def test_invalid_measurement_inputs_fail_closed(audio, rate):
    with pytest.raises(ValueError):
        measure_source(audio, rate)
