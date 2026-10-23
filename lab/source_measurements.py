"""Versioned measurements on decoded source channels, before mixing/resampling.

Sample peaks are not reconstructed true peaks. Near-full-scale occupancy is an
amplitude observation, not a diagnosis of clipping or an authenticity detector.
"""
import math

import numpy as np

VERSION = "source-measurements-1.0"
NEAR_FULL_SCALE = .999


def level_dbfs(amplitude):
    """Amplitude 1 is 0 dBFS; JSON null represents the undefined log of silence."""
    return 20 * math.log10(amplitude) if amplitude > 0 else None


def measure_source(audio, rate):
    x = np.asarray(audio)
    if (x.ndim != 2 or not len(x) or not 1 <= x.shape[1] <= 8
            or isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0):
        raise ValueError("Expected non-empty samples-by-channels audio and a positive integer sample rate")
    channels = x.shape[1]
    sums, squares = np.zeros(channels), np.zeros(channels)
    minima, maxima = np.full(channels, np.inf), np.full(channels, -np.inf)
    near, over = np.zeros(channels, dtype=np.int64), np.zeros(channels, dtype=np.int64)
    mix_squares = 0.0
    # Bound temporary allocations independently of recording length.
    for start in range(0, len(x), 65536):
        block = np.asarray(x[start:start + 65536], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Source measurements require finite samples")
        sums += block.sum(axis=0)
        squares += np.square(block).sum(axis=0)
        minima = np.minimum(minima, block.min(axis=0))
        maxima = np.maximum(maxima, block.max(axis=0))
        near += (np.abs(block) >= NEAR_FULL_SCALE).sum(axis=0)
        over += (np.abs(block) > 1).sum(axis=0)
        mix_squares += float(np.square(block.mean(axis=1)).sum())
    rms = np.sqrt(squares / len(x))
    reference_power = float(squares.sum() / (len(x) * channels))
    mix_power = mix_squares / len(x)
    ratio = min(1.0, max(0.0, mix_power / reference_power)) if reference_power > 0 else None
    return {
        "version": VERSION, "representation": "decoded source channels; no mixing, resampling or normalization",
        "sample_rate_hz": int(rate), "frames": len(x), "duration_seconds": len(x) / int(rate),
        "amplitude_reference": 1.0, "near_full_scale_threshold": NEAR_FULL_SCALE,
        "units": {"rms_dbfs": "dBFS", "sample_peak_dbfs": "dBFS", "crest_factor_db": "dB",
                  "dc_offset": "normalized amplitude", "near_full_scale_fraction": "fraction of channel samples",
                  "over_full_scale_fraction": "fraction of channel samples", "attenuation_db": "dB"},
        "channels": [{
            "channel": i + 1, "rms_amplitude": float(rms[i]), "rms_dbfs": level_dbfs(float(rms[i])),
            "minimum_sample": float(minima[i]), "maximum_sample": float(maxima[i]),
            "sample_peak_amplitude": float(max(abs(minima[i]), abs(maxima[i]))),
            "sample_peak_dbfs": level_dbfs(float(max(abs(minima[i]), abs(maxima[i])))),
            "crest_factor_db": level_dbfs(float(max(abs(minima[i]), abs(maxima[i])) / rms[i])) if rms[i] > 0 else None,
            "dc_offset": float(sums[i] / len(x)),
            "near_full_scale_samples": int(near[i]), "near_full_scale_fraction": float(near[i] / len(x)),
            "over_full_scale_samples": int(over[i]), "over_full_scale_fraction": float(over[i] / len(x)),
        } for i in range(channels)],
        "arithmetic_mixdown": {
            "reference": "mean original-channel power before resampling",
            "energy_ratio": ratio,
            "attenuation_db": -10 * math.log10(ratio) if ratio is not None and ratio > 0 else None,
            "complete_cancellation": reference_power > 0 and mix_power == 0,
        },
        "limitations": [
            "Measures decoded samples; decoder precision and lossy-codec history remain relevant.",
            "Sample peak is not oversampled true peak; near-full-scale samples do not prove clipping.",
            "Null dB values indicate zero energy or an undefined ratio, not missing samples.",
            "Mixdown attenuation describes channel interaction, not tampering or synthetic speech.",
        ],
    }
