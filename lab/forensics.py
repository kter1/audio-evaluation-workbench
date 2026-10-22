"""Local forensic measurements and bounded, reproducible detector inference.

Acoustic descriptors are observations, never an AI-origin classifier.
"""
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import BytesIO
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import platform
import time

import numpy as np
import soundfile as sf
from scipy import signal

from .audio import decode

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/models/wav2vec2-detector"
MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
REVISION = "c66306024a7ede0be291e9c4558b37634782dc4e"
MODEL_FILES = {
    "config.json": "c5791d2fc2e5506b228fa99387c2b2619c7836ad7e244699f8e5e8510b8f2574",
    "preprocessor_config.json": "35818933f1836450939a59d1f0ae7e4386d0d64d1560ce60a65001af62ac3f71",
    "README.md": "066e8ca93bae18ab2ece911f95d3bad640f4516c43f924aeda66ba6352c2bc64",
    "model.safetensors": "905e330265c40a76955911ce86691fb4c8f2ed9c8085f4f046cac1383592a0ac",
}
RATE = 16000
MAX_SECONDS = 600
MAX_BYTES = 256 * 1024 * 1024
POLICY = {"version": "1.0", "binary_threshold": 0.5, "human_below": 0.35,
          "synthetic_above": 0.65, "window_seconds": 6,
          "minimum_duration_seconds": 2.5, "minimum_rms_dbfs": -50,
          "minimum_active_fraction": 0.10, "maximum_clipped_fraction": 0.01,
          "status": "engineering abstention policy; not calibrated on a validation set"}
LIMITATIONS = ["English speech detector. Music, silence, other languages, mixed audio, unseen generators and codecs are outside established validation here.",
               "Local evaluation is limited to the disclosed benchmark; probability calibration is not established. Benchmark metrics are not results on your file.",
               "The model classifies acoustic patterns; it cannot establish authorship, intent, generator identity, or prove authenticity.",
               "Spectral and signal-quality descriptors are not model explanations or independent evidence of AI origin.",
               "Window scores do not establish exact edited boundaries; perturbation tests are sensitivity probes, not a second detector."]


def abstention_band(threshold, logit_width=None):
    """Engineering band around a detector's threshold: ±0.15 in probability, or ±width in logits.

    The logit form keeps the band inside (0, 1) for thresholds near 1, such as the v4 fusion's.
    """
    if logit_width is None:
        return round(threshold - .15, 10), round(threshold + .15, 10)
    centre = np.log(threshold / (1 - threshold))
    return float(1 / (1 + np.exp(-(centre - logit_width)))), float(1 / (1 + np.exp(-(centre + logit_width))))


def fmt_p(value):
    """Probabilities near 0 or 1 need more digits to stay distinguishable."""
    return f"{value:.2f}" if .01 <= value <= .99 else f"{value:.4f}"


def read_audio(blob):
    if not blob or len(blob) > MAX_BYTES:
        raise ValueError("Choose a non-empty audio file of at most 50 MiB.")
    try:
        info = sf.info(BytesIO(blob))
        if info.channels > 8 or not 8000 <= info.samplerate <= 192000:
            raise ValueError("Supported input: 1–8 channels and 8–192 kHz sample rate.")
        if not 0.5 <= info.duration <= MAX_SECONDS:
            raise ValueError("Choose a recording between 0.5 and 600 seconds.")
        # Limit allocation even for malformed headers.
        if info.frames * info.channels > 60_000_000:
            raise ValueError("Decoded audio exceeds the 60-million-sample limit.")
        audio, rate = decode(BytesIO(blob))
        if len(audio) != info.frames or rate != info.samplerate or audio.shape[1] != info.channels:
            raise ValueError("Decoded audio differs from its declared format.")
        return audio, rate, {"format": info.format, "subtype": info.subtype,
                             "original_rate": rate, "channels": info.channels,
                             "frames": len(audio), "duration_seconds": len(audio) / rate,
                             "bytes": len(blob), "sha256": sha256(blob).hexdigest()}
    except (RuntimeError, OSError) as exc:
        raise ValueError("This file could not be decoded. Try PCM WAV, FLAC, MP3, OGG, or AIFF.") from exc


def mono16(audio, rate):
    if audio.ndim != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected finite, non-empty audio with channels.")
    mono = audio.astype(np.float64).mean(axis=1)
    if rate != RATE:
        factor = gcd(rate, RATE)
        mono = signal.resample_poly(mono, RATE // factor, rate // factor)
    return mono.astype(np.float32)


def dbfs(value):
    return float(20 * np.log10(max(float(value), 1e-12)))


def diagnostics(audio, rate):
    x = mono16(audio, rate)
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    peak = float(np.max(np.abs(x)))
    size = 320  # 20 ms, no zero padding used for activity statistics
    frames = x[:len(x)//size*size].reshape(-1, size)
    levels = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
    activity = float(np.mean(levels > 10**(-50/20)))
    freqs, psd = signal.welch(x, RATE, window="hann", nperseg=512, noverlap=256)
    total = float(psd.sum())
    normalized = psd / max(total, 1e-30)
    centroid = float(np.sum(freqs * normalized))
    rolloff = float(freqs[min(np.searchsorted(np.cumsum(normalized), .95), len(freqs)-1)]) if total > 0 else None
    flatness = float(np.exp(np.mean(np.log(np.maximum(psd, 1e-30)))) / max(float(psd.mean()), 1e-30)) if total else None
    clipping = float(np.mean(np.abs(audio) >= .999))
    channel_rms = [dbfs(np.sqrt(np.mean(audio[:, c].astype(np.float64)**2))) for c in range(audio.shape[1])]
    correlation = None
    if audio.shape[1] == 2 and all(np.std(audio[:, c]) > 1e-8 for c in range(2)):
        correlation = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
    warnings = []
    if len(x)/RATE < 2.5:
        warnings.append("Below the lab's 2.5-second minimum-context policy.")
    if rms < 10**(-50/20) or activity < .10:
        warnings.append("Insufficient active signal; origin inference is inconclusive.")
    if clipping > .01:
        warnings.append("More than 1% of channel samples are at full scale; clipping can distort model evidence.")
    if rate < RATE:
        warnings.append("Upsampling cannot restore frequencies missing from the source.")
    if correlation is not None and correlation < -.5:
        warnings.append("Opposite-phase stereo may cancel speech when mixed to mono.")
    if peak > 1:
        warnings.append("Floating-point audio exceeds full scale; amplitude is preserved for analysis.")
    metrics = {"rms_dbfs": dbfs(rms), "peak_dbfs": dbfs(peak),
               "crest_factor_db": dbfs(peak/max(rms, 1e-12)), "dc_offset": float(x.mean()),
               "clipped_sample_fraction": clipping, "active_frame_fraction": activity,
               "activity_definition": "20 ms frame RMS > -50 dBFS; energy activity, not speech detection",
               "channel_rms_dbfs": channel_rms, "stereo_correlation": correlation,
               "spectral_centroid_hz": centroid, "rolloff_95_hz": rolloff,
               "spectral_flatness": flatness, "zero_crossing_rate": float(np.mean(np.diff(np.signbit(x)))),
               "band_power_fractions": {"0–300 Hz": float(normalized[freqs < 300].sum()),
                                        "300–3400 Hz": float(normalized[(freqs >= 300) & (freqs < 3400)].sum()),
                                        "3400–8000 Hz": float(normalized[freqs >= 3400].sum())}}
    return x, metrics, warnings


def windows(x, size=6 * RATE, merge_short_tail=True):
    cuts = list(range(0, len(x), size))
    if merge_short_tail and len(cuts) > 1 and len(x) - cuts[-1] < int(2.5 * RATE):
        cuts.pop()  # merge a short tail; maximum window is < 8.5 s
    return [(start, cuts[i+1] if i+1 < len(cuts) else len(x)) for i, start in enumerate(cuts)]


def anchored_spans(n, size):
    """Complete windows; a remaining tail gets a full window anchored at the end."""
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n-size+1, size))
    if starts[-1]+size < n:
        starts.append(n-size)
    return [(a, a+size) for a in starts]


def decision(scores, metrics, duration, threshold=POLICY["binary_threshold"], band=None):
    if not scores:
        raise ValueError("No inference windows.")
    weight = np.array([s.get("aggregation_weight", s["end"]-s["start"]) for s in scores])
    values = np.array([s["synthetic_score"] for s in scores])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)) or np.any(weight <= 0):
        raise ValueError("Invalid detector output.")
    low, high = band or abstention_band(threshold)
    mean = float(np.average(values, weights=weight))
    binary = "synthetic" if mean >= threshold else "human"
    reasons = []
    if duration < POLICY["minimum_duration_seconds"]:
        reasons.append("Input is shorter than 2.5 seconds.")
    if metrics["rms_dbfs"] < POLICY["minimum_rms_dbfs"] or metrics["active_frame_fraction"] < POLICY["minimum_active_fraction"]:
        reasons.append("Too little active signal.")
    if metrics["clipped_sample_fraction"] > POLICY["maximum_clipped_fraction"]:
        reasons.append("Severe full-scale clipping.")
    if metrics.get("stereo_correlation") is not None and metrics["stereo_correlation"] < -.5:
        reasons.append("Stereo cancellation risk.")
    if values.min() < low and values.max() > high:
        reasons.append("Windows disagree; inspect temporal scores.")
    if low <= mean <= high:
        reasons.append("Aggregate score falls in the abstention interval.")
    return {"synthetic_score": mean, "binary_model_decision": binary,
            "assessment": "Inconclusive" if reasons else ("Synthetic speech indicated" if binary == "synthetic" else "Human speech indicated"),
            "abstention_reasons": reasons, "window_min": float(values.min()), "window_max": float(values.max()),
            "aggregation": "duration-weighted mean of non-overlapping window softmax scores",
            "interpretation": "Uncalibrated model score, not a probability that the recording is AI. Applies to speech; speech presence and language are not independently verified."}


def verify_model(directory=MODEL_DIR):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["repository"] != MODEL_ID or manifest["revision"] != REVISION:
            raise ValueError("Unexpected detector identity.")
        for name in ("config.json", "preprocessor_config.json", "model.safetensors", "README.md"):
            path = directory / name
            expected = manifest["files"][name]
            h = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    h.update(chunk)
            if (path.stat().st_size != expected["bytes"] or h.hexdigest() != expected["sha256"]
                    or expected["sha256"] != MODEL_FILES[name]):
                raise ValueError(f"Model file failed checksum verification: {name}")
        return manifest
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Detector is not installed. Run .venv/bin/python scripts/setup_detector.py") from exc


class Detector:
    name = "Wav2Vec2-XLSR"
    description = "24 transformer layers · English speech detector"
    training_domain = ("English human recordings and six TTS services, 1,866 clips, 2.5–13 seconds. "
                       "Local comparison exposed frequent human false positives at the default threshold.")

    def __init__(self, directory=MODEL_DIR):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.manifest = verify_model(directory)
        torch.set_num_threads(min(4, max(1, __import__('os').cpu_count() or 1)))
        self.torch = torch
        # CPU is deterministic and avoids device-dependent MPS compatibility issues.
        self.extractor = AutoFeatureExtractor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForAudioClassification.from_pretrained(str(directory), local_files_only=True,
                                                                    trust_remote_code=False, use_safetensors=True).eval()
        if self.model.config.id2label != {0: "real", 1: "fake"} or self.extractor.sampling_rate != RATE:
            raise ValueError("Detector label mapping or input rate differs from the pinned contract.")
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def score(self, x):
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("Cannot infer on empty or non-finite audio.")
        inputs = self.extractor(x, sampling_rate=RATE, return_tensors="pt", padding=False)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1)
        result = {"synthetic_score": float(probabilities[1]), "human_score": float(probabilities[0]),
                  "logits": [float(v) for v in logits]}
        if not np.isfinite(result["logits"]).all():
            raise ValueError("Detector returned non-finite logits.")
        return result


def analyze(blob, detector, robustness=False, on_progress=None):
    started = time.perf_counter()
    audio, rate, source = read_audio(blob)
    x, metrics, warnings = diagnostics(audio, rate)
    spans = detector.spans(x) if hasattr(detector, "spans") else windows(x)
    scores = []
    for i, (a, b) in enumerate(spans):
        score = detector.score(x[a:b])
        scores.append({"start": a/RATE, "end": b/RATE,
                       "aggregation_weight": 1.0 if hasattr(detector, "spans") else (b-a)/RATE, **score})
        if on_progress:
            on_progress((i+1)/len(spans), f"Scored window {i+1} of {len(spans)}")
    threshold = getattr(detector, "threshold", POLICY["binary_threshold"])
    low, high = abstention_band(threshold, getattr(detector, "band_logit", None))
    assessment = decision(scores, metrics, len(x)/RATE, threshold, (low, high))
    if hasattr(detector, "spans"):
        assessment["aggregation"] = detector.preprocessing["aggregation"]
    stability = []
    if robustness:
        # All perturbations share the most energetic window and baseline.
        index = max(range(len(spans)), key=lambda i: float(np.mean(x[spans[i][0]:spans[i][1]]**2)))
        a, b = spans[index]
        sample = x[a:b]
        noise = np.random.default_rng(20261019).normal(0, 1, len(sample))
        noise *= np.sqrt(np.mean(sample.astype(np.float64)**2)) / max(np.sqrt(np.mean(noise**2)), 1e-12) * 10**(-30/20)
        lowpass = signal.sosfiltfilt(signal.butter(6, 3400, fs=RATE, output="sos"), sample).astype(np.float32)
        for name, changed in [("Gain −6.02 dB", sample*.5), ("Low-pass 3.4 kHz / Butterworth order 6 / zero phase", lowpass),
                              ("Gaussian noise / 30 dB SNR / seed 20261019", (sample+noise).astype(np.float32))]:
            result = detector.score(changed)
            stability.append({"transformation": name, "start": a/RATE, "end": b/RATE,
                              "baseline_synthetic_score": scores[index]["synthetic_score"],
                              **result, "score_change": result["synthetic_score"]-scores[index]["synthetic_score"]})
        if max(abs(r["score_change"]) for r in stability) > .2:
            warnings.append("Model score changed by more than 0.20 under a tested perturbation. Stability is limited.")
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "scipy", "soundfile", "torch", "transformers", "safetensors")}
    report = {"schema_version": "forensics-1.1", "created_at": datetime.now(timezone.utc).isoformat(),
              "source": source, "model": {**detector.manifest, "parameters": detector.parameters,
              "name": getattr(detector, "name", "Wav2Vec2-XLSR"), "description": getattr(detector, "description", "Local classifier"),
              "class_mapping": getattr(detector, "class_mapping", {"0": "human / real", "1": "synthetic / fake"}),
              "training_domain": getattr(detector, "training_domain", None), "device": "cpu", "dtype": "float32"},
              "preprocessing": {"mono": "arithmetic channel mean", "sample_rate": RATE,
              "resampling": "scipy.signal.resample_poly, reduced rational factors, default Kaiser filter",
              "normalization": "pinned feature extractor, separately per inference window", "padding": "none", "window_tail": "tails <2.5 s merged into preceding window", "window_samples": 6*RATE, **getattr(detector, "preprocessing", {})},
              "policy": {**POLICY, "binary_threshold": threshold, "human_below": low, "synthetic_above": high,
                         "window_policy_version": "2.0" if hasattr(detector, "spans") else "1.0", "window_seconds": getattr(detector, "window_samples", 6*RATE)/RATE}, "assessment": assessment, "diagnostics": metrics, "warnings": warnings,
              "windows": scores, "robustness": stability, "runtime_seconds": time.perf_counter()-started,
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
              "limitations": [*getattr(detector, "limitations", []), *LIMITATIONS],
              "sources": getattr(detector, "sources", [f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}", "https://huggingface.co/docs/transformers/model_doc/wav2vec2"])}
    return report, x


def explain(report):
    """The fixed checks behind an assessment, each measured value against its limit."""
    a, m, policy = report["assessment"], report["diagnostics"], report["policy"]
    threshold, low, high = policy["binary_threshold"], policy["human_below"], policy["synthetic_above"]
    score, duration = a["synthetic_score"], report["source"]["duration_seconds"]
    lo = min(report["windows"], key=lambda w: w["synthetic_score"])
    hi = max(report["windows"], key=lambda w: w["synthetic_score"])

    def check(name, measured, limit, ok):
        return {"check": name, "measured": measured, "limit": limit,
                "status": "n/a" if ok is None else ("pass" if ok else "fail")}

    span = lambda w: f'{fmt_p(w["synthetic_score"])} at {w["start"]:.2f}–{w["end"]:.2f} s'
    correlation = m.get("stereo_correlation")
    shift = max((abs(r["score_change"]) for r in report["robustness"]), default=None)
    checks = [
        check("Score clear of the uncertain band", fmt_p(score), f"outside {fmt_p(low)}–{fmt_p(high)}", not low <= score <= high),
        check("Windows agree", f"lowest {span(lo)}; highest {span(hi)}", f"not both below {fmt_p(low)} and above {fmt_p(high)}",
              not (lo["synthetic_score"] < low and hi["synthetic_score"] > high)),
        check("Long enough", f"{duration:.2f} s", f"≥ {policy['minimum_duration_seconds']:.2f} s", duration >= policy["minimum_duration_seconds"]),
        check("Loud enough", f"{m['rms_dbfs']:.1f} dBFS", f"≥ {policy['minimum_rms_dbfs']} dBFS", m["rms_dbfs"] >= policy["minimum_rms_dbfs"]),
        check("Enough active signal", f"{m['active_frame_fraction']:.1%}", f"≥ {policy['minimum_active_fraction']:.0%}",
              m["active_frame_fraction"] >= policy["minimum_active_fraction"]),
        check("No severe clipping", f"{m['clipped_sample_fraction']:.2%}", f"≤ {policy['maximum_clipped_fraction']:.0%}",
              m["clipped_sample_fraction"] <= policy["maximum_clipped_fraction"]),
        check("No stereo cancellation", "mono" if correlation is None else f"{correlation:.2f}", "correlation ≥ −0.50",
              None if correlation is None else correlation >= -.5),
        check("Stable under perturbations", "not run" if shift is None else f"largest change {shift:.3f}", "≤ 0.20",
              None if shift is None else shift <= .2),
    ]
    synthetic = a["binary_model_decision"] == "synthetic"
    applicable = [c for c in checks if c["status"] != "n/a"]
    failing = [c["check"].lower() for c in applicable if c["status"] == "fail"]
    if a["assessment"] == "Inconclusive":
        verdict = (f"Inconclusive: the model leans {a['binary_model_decision']} (score {fmt_p(score)} vs threshold {fmt_p(threshold)}), "
                   "but " + "; ".join(r.rstrip(".").lower() for r in a["abstention_reasons"]) + ".")
    else:
        verdict = (f"{'Flagged as synthetic' if synthetic else 'Judged human'}: score {fmt_p(score)} is "
                   f"{'at or above' if synthetic else 'below'} the {fmt_p(threshold)} threshold")
        verdict += (f" and all {len(applicable)} applicable reliability checks passed." if not failing
                    else f". Caution: {', '.join(failing)} failed.")
    components = {}
    for w in report["windows"]:
        for name, value in w.get("components", {}).items():
            components.setdefault(name, []).append(value)
    return {"verdict": verdict, "checks": checks, "components": {k: float(np.mean(v)) for k, v in components.items()}}


def save_report(report, root=ROOT / "data/analyses"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    # Content-derived names preserve repeated runs with different settings/results.
    path = root / (sha256(text.encode()).hexdigest() + ".json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text + "\n")
    temporary.replace(path)
    return path


def report_html(report):
    def table(headers, rows):
        head = ''.join(f'<th>{escape(str(h))}</th>' for h in headers)
        body = ''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)
        return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    a, source = report['assessment'], report['source']
    duration = source['duration_seconds']
    policy = report['policy']
    threshold, low, high = policy['binary_threshold'], policy['human_below'], policy['synthetic_above']
    why = explain(report)
    mark = {'pass': '✓ pass', 'fail': '✗ fail', 'n/a': '— n/a'}
    why_table = table(['Check', 'Measured', 'Limit', 'Result'], [[c['check'], c['measured'], c['limit'], mark[c['status']]] for c in why['checks']])
    segments = ''.join(f'<line x1="{40+720*r["start"]/duration:.2f}" x2="{40+720*r["end"]/duration:.2f}" y1="{170-140*r["synthetic_score"]:.2f}" y2="{170-140*r["synthetic_score"]:.2f}" stroke="#b9563e" stroke-width="4"/>' for r in report['windows'])
    plot = f'<svg viewBox="0 0 800 210" role="img" aria-label="Synthetic class score over time"><rect x="40" y="30" width="720" height="140" fill="#f1f4f8"/><rect x="40" y="{170-140*high:.2f}" width="720" height="{140*(high-low):.2f}" fill="#d8e0ea"/>{segments}<line x1="40" x2="760" y1="{170-140*threshold:.2f}" y2="{170-140*threshold:.2f}" stroke="#8795a7" stroke-dasharray="4 4"/><g fill="#536980" font-family="monospace" font-size="12"><text x="14" y="35">1</text><text x="14" y="173">0</text><text x="40" y="195">0 s</text><text x="700" y="195">{duration:.2f} s</text></g></svg>'
    windows_table = table(['Start s','End s','Synthetic score','Human score','Human logit','Synthetic logit'],
        [[f'{r["start"]:.3f}', f'{r["end"]:.3f}', f'{r["synthetic_score"]:.6f}', f'{r["human_score"]:.6f}', f'{r["logits"][0]:.6f}', f'{r["logits"][1]:.6f}'] for r in report['windows']])
    sensitivity_table = table(['Perturbation','Interval s','Baseline','Changed score','Change'],
        [[r['transformation'],f'{r["start"]:.2f}–{r["end"]:.2f}', f'{r["baseline_synthetic_score"]:.6f}',f'{r["synthetic_score"]:.6f}',f'{r["score_change"]:+.6f}'] for r in report['robustness']]) if report['robustness'] else '<p>Not requested.</p>'
    alerts = ''.join(f'<li>{escape(s)}</li>' for s in a['abstention_reasons']+report['warnings'])
    limitations = ''.join(f'<li>{escape(s)}</li>' for s in report['limitations'])
    details = ''.join(f'<h3>{escape(k.title())}</h3><pre>{escape(json.dumps(report[k], indent=2))}</pre>' for k in ('source','diagnostics','model','preprocessing','policy','environment','sources'))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Audio Forensics Report</title>
<style>body{{font:14px system-ui;max-width:1020px;margin:48px auto;color:#19283c;padding:20px;line-height:1.6}}.eyebrow{{color:#a0553c;letter-spacing:2px;font:11px monospace}}h1{{font-size:32px;letter-spacing:-1px}}h2{{margin-top:36px;font-size:21px}}h3{{font-size:15px}}.hero{{background:#182b43;color:white;padding:26px;border-radius:16px}}.hero h1{{margin:0}}.hero p{{color:#c9d7e6}}.cards{{display:flex;gap:16px;margin:20px 0}}.card{{flex:1;border:1px solid #dce3eb;padding:18px;border-radius:12px}}.number{{font:26px monospace;font-weight:bold}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:18px;border-radius:12px;font-size:11px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{text-align:left;border-bottom:1px solid #e2e8ee;padding:10px}}th{{background:#f0f4f8}}.table{{overflow:auto}}.notes{{background:#fff4e8;padding:16px 32px;border-radius:12px}}.muted{{color:#61758c;font-size:12px}}@media print{{body{{margin:0;font-size:11px}}.hero{{color:#19283c;background:#eef2f7}}.hero p{{color:#52677f}}h2,h3{{break-after:avoid}}tr,.cards,svg{{break-inside:avoid}}pre{{font-size:9px}}}}</style></head><body>
<p class="eyebrow">AUDIO FORENSICS LAB / TECHNICAL ANALYSIS</p>
<div class="hero"><h1>{escape(a['assessment'])}</h1><p>Fresh local neural inference · {escape(report['created_at'])}</p></div>
<div class="cards"><div class="card">Synthetic-class score<div class="number">{a['synthetic_score']:.6f}</div></div><div class="card">Duration<div class="number">{duration:.2f} s</div></div><div class="card">Windows<div class="number">{len(report['windows'])}</div></div></div>
<p>{escape(a['interpretation'])}</p>
<p class="muted">SHA-256: {escape(source['sha256'])}<br>Checkpoint: {escape(report['model']['repository'])} / {escape(report['model']['revision'])}</p>
<h2>00 / Why this result</h2><p><b>{escape(why['verdict'])}</b></p>{why_table}
<h2>01 / Reliability observations</h2><ul class="notes">{alerts or '<li>No configured quality warning triggered. Correctness remains unverified.</li>'}</ul>
<h2>02 / Temporal detector evidence</h2>{plot}{windows_table}
<p class="muted">{escape(a['aggregation'])}. Grey band: engineering abstention interval [{fmt_p(low)}, {fmt_p(high)}]; dotted line: raw binary threshold {fmt_p(threshold)}. Neither is a calibrated confidence interval.</p>
<h2>03 / Controlled sensitivity probes</h2>{sensitivity_table}<p class="muted">Shared most energetic window; three separate transformations. These tests measure sensitivity, not accuracy.</p>
<h2>04 / Scope and interpretation</h2><ul>{limitations}</ul>
<h2>05 / Reproduction record and signal measurements</h2>{details}
<p class="muted">Schema {escape(report['schema_version'])} · Computation {report['runtime_seconds']:.3f} s · Audio retained by user; this report contains measurements and provenance.</p></body></html>'''
