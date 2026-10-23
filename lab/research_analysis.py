"""Experimental methods, isolated from production findings and classifier inputs."""
import importlib.metadata
import math

import numpy as np
from scipy.stats import chi2

from lab.signal_events import acoustic_markers
from lab.source_measurements import level_dbfs

RATE, HOP = 16000, 320
CONFIG = {
    "version": "analysis-research-1.0", "rate_hz": RATE, "frame_seconds": .02,
    "energy_threshold_dbfs": -50., "noise_quantile": .1,
    "rolling_noise_block_seconds": 2., "change_side_seconds": .12,
    "change_context_seconds": 2., "change_minimum_db": 6., "change_mad_multiplier": 6.,
    "change_minimum_side_dbfs": -50., "change_separation_seconds": .3,
    "webrtc_mode": 2, "promotion_status": "experimental; no production integration",
    "gaussian_noise_correction": "divide frame-power q10 by chi2.ppf(0.1, 320)/320; assumes iid zero-mean Gaussian noise",
}


def frame_power(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or len(x) < RATE // 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least 0.5 s of finite mono 16 kHz samples")
    frames = x[:len(x) // HOP * HOP].reshape(-1, HOP)
    return np.mean(frames**2, axis=1)


def intervals_from_mask(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    return [{"start": float(a * HOP / RATE), "end": float(b * HOP / RATE)}
            for a, b in zip(changes[::2], changes[1::2])]


def energy_activity(samples):
    """Existing energy activity used as a diagnostic baseline, never called VAD."""
    return intervals_from_mask(frame_power(samples) > 10**(CONFIG["energy_threshold_dbfs"] / 10))


def noise_estimates(samples):
    """Two lower-tail power statistics; neither is a validated speech/noise separator."""
    power = frame_power(samples)
    q = CONFIG["noise_quantile"]
    block = round(CONFIG["rolling_noise_block_seconds"] * RATE / HOP)
    rolling = [float(np.quantile(power[i:i + block], q)) for i in range(0, len(power), block)]
    lower = float(np.quantile(power, q))
    # For iid zero-mean Gaussian samples, N*frame_power/sigma^2 ~ chi2(N).
    # This is an assumption-bound analytic correction, not a learned SNR model.
    correction = float(chi2.ppf(q, HOP)/HOP)
    return {"global_lower_decile": level_dbfs(math.sqrt(lower)),
            "blockwise_lower_decile_median": level_dbfs(math.sqrt(float(np.median(rolling)))),
            "iid_gaussian_corrected_decile": level_dbfs(math.sqrt(lower/correction))}


def level_changes(samples, method):
    if method == "frozen_level_rule":
        return [{"start": m["start"], "end": m["end"]} for m in acoustic_markers(samples, RATE, "research")
                if m["kind"] == "level_change"]
    if method != "adaptive_level_rule":
        raise ValueError("Unknown level-change method")
    power = frame_power(samples)
    levels = 10 * np.log10(np.maximum(power, 1e-24))
    side = round(CONFIG["change_side_seconds"] * RATE / HOP)
    context = round(CONFIG["change_context_seconds"] * RATE / HOP)
    candidates = []
    for i in range(side, len(power) - side + 1):
        left = 10 * np.log10(max(float(power[i-side:i].mean()), 1e-24))
        right = 10 * np.log10(max(float(power[i:i+side].mean()), 1e-24))
        local_differences = np.diff(levels[max(0, i-context):min(len(levels), i+context)])
        median = np.median(local_differences)
        spread = 1.4826 * float(np.median(np.abs(local_differences - median)))
        limit = max(CONFIG["change_minimum_db"], CONFIG["change_mad_multiplier"] * spread)
        if min(left, right) >= CONFIG["change_minimum_side_dbfs"] and abs(right - left) >= limit:
            candidates.append((i, abs(right-left)))
    separation = round(CONFIG["change_separation_seconds"] * RATE / HOP)
    selected = []
    for i, strength in sorted(candidates, key=lambda p: (-p[1], p[0])):
        if all(abs(i-j) >= separation for j in selected):
            selected.append(i)
    return [{"start": (i-side)*HOP/RATE, "end": (i+side)*HOP/RATE} for i in sorted(selected)]


def optional_webrtc(samples):
    """Explicit PCM conversion for the optional VAD only; no download or setup."""
    frame_power(samples)  # same validation/coverage policy as the energy baseline
    try:
        import webrtcvad
    except ImportError:
        return {"status": "unavailable", "reason": "webrtcvad is not installed; no speech scores fabricated"}
    version = None
    for distribution in ("webrtcvad", "webrtcvad-wheels"):
        try:
            version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if version is None:
        return {"status": "unavailable", "reason": "VAD module exists but has no traceable package version"}
    vad = webrtcvad.Vad(CONFIG["webrtc_mode"])
    x = np.asarray(samples)
    pcm = np.rint(np.clip(x, -1, 32767/32768)*32768).astype("<i2")
    mask = [vad.is_speech(pcm[i:i+HOP].tobytes(), RATE) for i in range(0, len(pcm)-HOP+1, HOP)]
    return {"status": "available", "distribution": distribution, "package_version": version,
            "intervals": intervals_from_mask(mask), "input_conversion": "mono16 -> clipped, rounded signed little-endian PCM16; no padded tail",
            "clipped_input_samples": int(np.count_nonzero((x < -1) | (x > 32767/32768)))}
