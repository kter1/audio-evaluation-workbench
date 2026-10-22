"""Additional bounded audio-quality observations for SignalTrace.

These checks are deterministic engineering rules, not AI-speech-origin detectors.
The rules deliberately distinguish measurement from causal interpretation.
"""
from __future__ import annotations

import numpy as np

from .signal_events import RATE, HOP, _db, _intervals, marker

ANALYSIS_VERSION = "signal-quality-1.1"


def extra_acoustic_markers(samples, source_hash):
    """Detect gaps, isolated impulses, DC offsets and bandwidth restrictions."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or len(x) < RATE // 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite mono 16 kHz audio.")
    duration = len(x) / RATE
    count = len(x) // HOP
    frames = x[:count * HOP].reshape(-1, HOP).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    levels = 20 * np.log10(np.maximum(rms, 1e-10))
    result = []

    # Short drop in signal energy, bounded by relatively energetic audio on
    # both sides. Distinct from broad silence/pause intervals.
    quiet = np.flatnonzero(levels < -55)
    for a, b in _intervals(quiet, gap=0):
        seconds = (b - a) * .02
        if seconds < .08 or seconds > .34 or a < 8 or b + 8 > count:
            continue
        left = float(np.median(levels[a-8:a]))
        right = float(np.median(levels[b:b+8]))
        if left < -30 or right < -30:
            continue
        result.append(marker(
            source_hash, "signal_gap", "signal", a * .02, b * .02,
            "Short signal gap",
            "A brief low-energy span is surrounded by markedly stronger audio.",
            {"gap_ms": round(seconds * 1000), "preceding_dbfs": round(left, 1),
             "following_dbfs": round(right, 1)},
            "20 ms RMS frames; compare 160 ms before and after",
            "80–340 ms below -55 dBFS, both flanks above -30 dBFS",
            "It may be a pause, gate, packet loss or editing; the cause cannot be inferred.",
            "Listen before and after this gap and check acquisition logs."))

    # A click-like digital discontinuity has a large neighboring-sample jump
    # in otherwise low/moderate-energy local audio. Avoid calling a musical
    # transient or speech plosive a confirmed digital glitch.
    candidates = np.flatnonzero(np.abs(np.diff(x)) >= .75)
    chosen = []
    for i in candidates:
        if len(chosen) >= 8:
            break
        if chosen and i - chosen[-1] < int(.35 * RATE):
            continue
        left = max(0, int(i)-int(.025*RATE))
        right = min(len(x), int(i)+int(.025*RATE))
        around = x[left:right]
        if not len(around):
            continue
        local_rms = float(np.sqrt(np.mean(around.astype(np.float64)**2)))
        if local_rms > .20:
            continue
        chosen.append(int(i))
        result.append(marker(
            source_hash, "impulse", "signal",
            max(0, (i-320)/RATE), min(duration, (i+320)/RATE),
            "Impulse-like discontinuity",
            "Neighboring audio samples exhibit an abrupt isolated amplitude jump.",
            {"sample_jump": round(float(abs(x[i+1]-x[i])), 4),
             "local_rms": round(local_rms, 4)},
            "Adjacent PCM sample differences, 50 ms surrounding energy",
            "Jump >= 0.75 full-scale with local RMS <= 0.20",
            "A click, digital splice, percussion, or other transient can produce this.",
            "Play the short interval at reduced volume and examine adjacent samples.",
            "high"))

    # 1-second windows to avoid reporting stable, harmless near-zero bias.
    width = RATE
    offset_candidates = []
    for start in range(0, len(x)-width+1, width//2):
        segment = x[start:start+width].astype(np.float64)
        mean = float(np.mean(segment))
        rms_value = float(np.sqrt(np.mean(segment * segment)))
        if rms_value >= .035 and abs(mean) >= .03 and abs(mean)/rms_value >= .45:
            offset_candidates.append((start, mean, rms_value))
    for start, mean, rms_value in offset_candidates[:5]:
        result.append(marker(
            source_hash, "dc_offset", "signal", start/RATE, (start+width)/RATE,
            "Waveform baseline offset",
            "The average sample value is displaced away from zero for this interval.",
            {"mean_amplitude": round(mean, 4), "rms_amplitude": round(rms_value, 4)},
            "One-second mean/DC-to-RMS ratio, 500 ms inspection hop",
            "|Mean| >= 0.03 and >= 45% of RMS (RMS >= 0.035)",
            "May arise from microphone or conversion artifacts; not evidence of synthetic speech.",
            "Inspect source hardware, encoding and waveform baseline."))

    # Detect an acoustic high-band roll-off as a signal property, not proof of
    # telephone encoding or artificial generation. Use half-second windows.
    size, step = RATE, RATE//2
    restricted = []
    for start in range(0, len(x)-size+1, step):
        chunk = x[start:start+size].astype(np.float64)
        level = float(np.sqrt(np.mean(chunk**2)))
        if level < .013:
            continue
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(size)))**2
        bands = spectrum[1:]
        total = float(bands.sum())
        if total <= 1e-12:
            continue
        frequencies = np.fft.rfftfreq(size, 1/RATE)
        high = float(spectrum[frequencies >= 3600].sum()/total)
        if high <= .002:
            restricted.append((start, high))
    # Adjacent windows may represent the same acoustic condition; group.
    if restricted:
        chosen = []
        for start, fraction in restricted:
            if not chosen or start > chosen[-1][-1] + step:
                chosen.append([start, start, fraction])
            else:
                chosen[-1][1] = start
                chosen[-1][2] = min(chosen[-1][2], fraction)
        for begin, last, fraction in chosen[:5]:
            end = min(duration, (last + size)/RATE)
            if end - begin/RATE < 1.0:
                continue
            result.append(marker(
                source_hash, "bandwidth_limit", "signal", begin/RATE, end,
                "Restricted high-frequency energy",
                "Very little signal energy is present above 3.6 kHz in this active span.",
                {"high_band_fraction": round(fraction, 6), "cutoff_hz": 3600},
                "Hann-window spectral power, 1-second windows, 0.5-second hop",
                "Above-3.6-kHz fraction <= 0.002 in active audio, lasting >= 1 s",
                "A low-pitched tone, codec, bandwidth limit or mic response may explain this.",
                "Compare frequency distribution against surrounding audio and recording settings."))

    return sorted(result, key=lambda m: (m["start"], m["kind"]))


def stereo_markers(audio, rate, source_hash):
    """Identify local stereo inversion and persistent channel imbalance."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2 or not 8000 <= rate <= 192000:
        return []
    interval = max(1, int(rate))
    result = []
    for start in range(0, len(x)-interval+1, interval):
        pair = x[start:start+interval].astype(np.float64)
        rms = np.sqrt(np.mean(pair**2, axis=0))
        if max(rms) < .01:
            continue
        small = float(min(rms))
        large = float(max(rms))
        difference = 20 * np.log10(max(large, 1e-9)/max(small, 1e-9))
        a, b = start/rate, (start+interval)/rate
        if difference >= 14:
            result.append(marker(
                source_hash, "channel_imbalance", "signal", a, b,
                "Stereo level imbalance",
                "One audio channel carries considerably more energy than the other.",
                {"left_rms_dbfs": round(_db(rms[0]), 1),
                 "right_rms_dbfs": round(_db(rms[1]), 1),
                 "difference_db": round(float(difference), 1)},
                "One-second per-channel RMS comparison",
                "Channel difference >= 14 dB",
                "May be intentional stereo panning, mono source placement or a recording fault.",
                "Solo each channel and inspect the original recording chain."))
        if min(rms) >= .015:
            centered = pair - pair.mean(axis=0, keepdims=True)
            std = centered.std(axis=0)
            if min(std) > 1e-6:
                correlation = float(np.mean(centered[:,0]*centered[:,1])/(std[0]*std[1]))
                if correlation <= -.75:
                    result.append(marker(
                        source_hash, "phase_inversion", "signal", a, b,
                        "Opposing stereo channels",
                        "Left and right channels have strong negative correlation.",
                        {"channel_correlation": round(correlation, 3)},
                        "One-second Pearson correlation between stereo channels",
                        "Correlation <= -0.75, both channel RMS >= 0.015",
                        "Summing out-of-phase signals can attenuate speech; phase effects are not proof of manipulation.",
                        "Compare the left and right playback channels independently.",
                        "high"))
    # Both stereo checks are diagnostic, not thousands of duplicate observations.
    return sorted(result, key=lambda m:(m["start"], m["kind"]))[:12]
