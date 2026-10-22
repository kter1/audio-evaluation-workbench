"""Timestamped acoustic and model evidence.

Markers are measured observations, NOT verified edits, fraud, or explanations of
what caused a neural network's classification. Boundaries retain the source
method's time resolution. No heavyweight models needed for this module.
"""
from __future__ import annotations
from hashlib import sha256
import json
import math
import numpy as np

RATE = 16000
HOP = 320
RULE_VERSION = "acoustic-1.0"


def _db(value):
    return 20.0 * math.log10(max(float(value), 1e-10))


def _intervals(indices, gap=1):
    if not len(indices):
        return []
    start = last = int(indices[0])
    result = []
    for index in indices[1:]:
        index = int(index)
        if index <= last + gap + 1:
            last = index
        else:
            result.append((start, last + 1))
            start = last = index
    result.append((start, last + 1))
    return result


def marker(source_hash, kind, category, start, end, title, description,
           measurements, method, threshold, limitations, next_check, priority="review"):
    start, end = round(float(start), 3), round(float(end), 3)
    basis = json.dumps([source_hash, RULE_VERSION, kind, start, end, measurements],
                       sort_keys=True, allow_nan=False)
    return {
        "id": sha256(basis.encode()).hexdigest()[:18],
        "start": start, "end": end,
        "kind": kind, "category": category, "priority": priority,
        "title": title, "description": description,
        "measurements": measurements, "method": method, "threshold": threshold,
        "limitations": limitations, "next_check": next_check,
        "rule_version": RULE_VERSION, "source": "measured",
    }


def acoustic_markers(samples, rate, source_hash):
    x = np.asarray(samples, dtype=np.float64)
    if rate != RATE or x.ndim != 1 or len(x) < RATE // 2 or not np.isfinite(x).all():
        raise ValueError("Expected 0.5 seconds or more of finite mono 16 kHz samples.")
    duration = len(x) / RATE
    frame_count = len(x) // HOP
    blocks = x[:frame_count * HOP].reshape(-1, HOP)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    levels = np.array([_db(v) for v in rms])
    near_limit = np.mean(np.abs(blocks) >= .995, axis=1)
    found = []

    # Sustained near-full-scale values, not isolated natural peaks.
    for first, last in _intervals(np.flatnonzero(near_limit >= .08), gap=2):
        start, end = first * .02, min(duration, last * .02)
        if end - start < .08:
            continue
        found.append(marker(source_hash, "clipping", "signal", start, end,
            "Possible saturation", "Many samples stay close to the digital amplitude limit.",
            {"near_limit_fraction": round(float(np.mean(near_limit[first:last])), 4)},
            "20 ms frames; fraction of |sample| >= 0.995",
            "At least 8% near-limit samples for at least 80 ms",
            "Heavy limiting can produce a similar pattern. Saturation does not establish an edit.",
            "Replay the region and inspect the original recording.", "high"))

    for first, last in _intervals(np.flatnonzero(levels <= -44), gap=1):
        start, end = first * .02, min(duration, last * .02)
        if end - start < .34:
            continue
        found.append(marker(source_hash, "low_energy", "signal", start, end,
            "Quiet interval", "Audio energy stays below the rule's threshold.",
            {"median_level_dbfs": round(float(np.median(levels[first:last])), 1)},
            "20 ms short-time root-mean-square level",
            "At most -44 dBFS for 340 ms or longer",
            "Normal pauses and silence can produce this. Not proof of missing speech.",
            "Listen with context before and after the interval."))

    # Compare 120 ms regions on either side; report only the strongest transitions.
    candidates = []
    for i in range(6, frame_count - 6):
        left, right = _db(np.mean(rms[i-6:i])), _db(np.mean(rms[i:i+6]))
        if min(left, right) >= -40 and abs(right-left) >= 12:
            candidates.append((i, right-left))
    selected = []
    for i, change in sorted(candidates, key=lambda p: -abs(p[1])):
        if any(abs(i-j) < 15 for j, _ in selected):
            continue
        selected.append((i, change))
        if len(selected) == 8:
            break
    for i, change in sorted(selected):
        found.append(marker(source_hash, "level_change", "signal",
            max(0, (i-6)*.02), min(duration, (i+6)*.02),
            "Sudden level change", "The mean level changes sharply across adjacent windows.",
            {"level_difference_db": round(float(change), 1)},
            "Comparison of adjacent 120 ms RMS windows",
            "At least 12 dB change with both sides above -40 dBFS",
            "This may reflect normal speech emphasis or microphone movement.",
            "Replay before and after the transition."))

    # Relative spectral power distribution; not a proof or ML attribution.
    fft_size, stride = 3200, 1600
    spectral = []
    for start in range(0, len(x)-fft_size+1, stride):
        segment = x[start:start+fft_size]
        if _db(np.sqrt(np.mean(segment**2))) < -38:
            spectral.append(None)
            continue
        power = np.abs(np.fft.rfft(segment*np.hanning(fft_size)))**2
        bins = np.array([power[a:b].sum() for a,b in
                         ((1,40),(40,100),(100,220),(220,550),(550,len(power)))])
        spectral.append(bins/max(float(bins.sum()), 1e-12))
    candidates = []
    for i in range(1, len(spectral)):
        if spectral[i] is not None and spectral[i-1] is not None:
            distance = float(np.abs(spectral[i]-spectral[i-1]).sum()/2)
            if distance >= .55:
                candidates.append((i, distance))
    selected = []
    for i, distance in sorted(candidates, key=lambda pair: -pair[1]):
        if any(abs(i-j) <= 4 for j,_ in selected):
            continue
        selected.append((i, distance))
        if len(selected) == 5:
            break
    for i, distance in sorted(selected):
        t = i * stride / RATE
        found.append(marker(source_hash, "spectral_change", "signal",
            max(0, t-.16), min(duration, t+.16),
            "Spectral transition", "The frequency-band distribution changes between adjacent frames.",
            {"band_distance": round(distance, 3)},
            "Hann-windowed 200 ms FFT, 100 ms hop, five normalized power bands",
            "Total-variation distance at least 0.55; both frames above -38 dBFS",
            "Consonants, pitch, music and acoustic changes can cause this; no edit is proved.",
            "Listen with context and compare neighboring frequency content."))
    return sorted(found, key=lambda row:(row["start"],row["kind"]))[:40]


def model_markers(report):
    """Never present a 4-second inference window as an exact suspicious phoneme."""
    source_hash = report["source"]["sha256"]
    threshold = float(report["policy"]["binary_threshold"])
    result = []
    for w in report["windows"]:
        start, end = float(w["start"]), float(w["end"])
        score = float(w["synthetic_score"])
        if score >= threshold:
            item = marker(source_hash, "synthetic_score", "model", start, end,
                "Elevated synthetic-class score",
                "The frozen fusion classifier scored this entire window above its threshold.",
                {"score": round(score,5), "threshold": round(threshold,5)},
                "Existing two-detector inference with frozen fusion weights",
                "Window synthetic-class score at or above the frozen threshold",
                "Model score is not calibrated authenticity probability, edit localization, or proof of AI generation.",
                "Inspect the component models and compare adjacent windows.")
            item["source"] = "fusion"
            result.append(item)
        scores = w.get("components", {})
        if "pellav2" in scores and "antideepfake" in scores:
            difference = abs(scores["pellav2"] - scores["antideepfake"])
            if difference >= .4:
                item = marker(source_hash, "component_separation", "model", start, end,
                    "Detector scores differ",
                    "The two models produced scores with substantially different magnitudes.",
                    {"pellav2": round(scores["pellav2"],5),
                     "antideepfake": round(scores["antideepfake"],5),
                     "absolute_difference": round(difference,5)},
                    "Difference between component synthetic-class scores",
                    "Absolute score difference of at least 0.40",
                    "The models' score scales may differ; magnitude difference is not necessarily categorical disagreement.",
                    "Review model limitations and calibration before interpreting.")
                item["source"] = "fusion"
                result.append(item)
    for test in report.get("robustness", []):
        if abs(test["score_change"]) > .2:
            item = marker(source_hash, "sensitivity", "reliability",
                test["start"], test["end"], "Result changes under transformation",
                "The selected window's score moves after a controlled audio modification.",
                {"transformation": test["transformation"],
                 "score_change": round(test["score_change"], 4)},
                "Existing bounded perturbation test on one selected window",
                "Absolute synthetic-class score change above 0.20",
                "A sensitivity probe is not an independent detector or proof of misclassification.",
                "Check whether the recording matches tested conditions.")
            item["source"] = "fusion"
            result.append(item)
    return sorted(result, key=lambda row:(row["start"],row["kind"]))
