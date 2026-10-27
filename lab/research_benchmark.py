"""Development-only comparisons; analytic fixtures cannot authorize promotion."""
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from lab.baseline_audit import checked_path
from lab.corpus_inventory import connected_groups
from lab.forensics import mono16, read_audio
from lab.research_analysis import CONFIG, HOP, RATE, energy_activity, level_changes, noise_estimates, optional_webrtc
from lab.research_metrics import count_metrics, events, paired_event_gain, speech_frames, validate_intervals

METHODS = ("frozen_level_rule", "adaptive_level_rule")


def analytic_cases():
    """Non-speech signal fixtures with known gain steps and noise realization.

    Gain variants share their latent-noise source group. No generated signal is
    described as human speech or a representative audio forgery.
    """
    cases = []
    n, boundary = 6 * RATE, 3 * RATE
    for seed in range(4):
        noise = np.random.default_rng(seed).normal(0, .02, n).astype(np.float32)
        source = f"analytic:noise:{seed}"
        common = {"source_group": source, "evidence_type": "analytic_fixture", "speech_intervals": []}
        cases.append({**common, "id": f"stationary-noise-{seed}", "samples": noise,
                      "level_events": [], "noise_reference_dbfs": float(20*np.log10(np.sqrt(np.mean(noise.astype(float)**2))))})
        for gain_db in (6, 12, 18):
            x = noise.copy()
            x[boundary:] *= 10**(gain_db/20)
            cases.append({**common, "id": f"gain-{gain_db}db-{seed}", "samples": x,
                          "level_events": [{"start": 2.88, "end": 3.12}], "noise_reference_dbfs": None})
        ramp = noise * np.linspace(1, 10, n).astype(np.float32)
        cases.append({**common, "id": f"gradual-gain-{seed}", "samples": ramp,
                      "level_events": [], "noise_reference_dbfs": None})
    for name, samples in (("silence", np.zeros(n, np.float32)),
                          ("steady-tone", (.2*np.sin(2*np.pi*1000*np.arange(n)/RATE)).astype(np.float32))):
        cases.append({"id": name, "source_group": "analytic:" + name, "evidence_type": "analytic_fixture",
                      "samples": samples, "level_events": [], "speech_intervals": [], "noise_reference_dbfs": None})
    return cases


def load_manifest(path, root):
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != "method-benchmark-1.0" or document.get("partition") != "development":
        raise ValueError("Only method-benchmark-1.0 development manifests can be scored by this runner")
    cases, ids = [], set()
    for row in document.get("cases", []):
        if row.get("partition", "development") != "development":
            raise ValueError("Locked/calibration cases cannot be scored by the development runner")
        for key in ("id", "source_group", "path", "sha256", "annotation_provenance"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing benchmark field: {key}")
        if row["id"] in ids:
            raise ValueError("Duplicate benchmark case ID")
        ids.add(row["id"])
        raw = checked_path(Path(root), row["path"]).read_bytes()
        if sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Audio hash mismatch: {row['id']}")
        audio, rate, _ = read_audio(raw)
        cases.append({**row, "samples": mono16(audio, rate), "evidence_type": "supplied_development_annotations"})
    if not cases:
        raise ValueError("Benchmark requires cases")
    return cases


def run(cases, repetitions=10000, silero_model=None):
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Benchmark requires uniquely identified cases")
    silero_session = None
    if silero_model is not None:
        from lab.research_vad import SileroSession
        silero_session = SileroSession(silero_model)
    results = []
    input_identities = []
    totals = {method: np.zeros(3, dtype=int) for method in METHODS}
    negative_fp = {method: 0 for method in METHODS}
    negative_seconds = 0.
    annotated_event_cases = 0
    for case in cases:
        if not case.get("source_group"):
            raise ValueError("Benchmark requires source grouping")
        x = np.asarray(case["samples"], dtype=np.float32)
        duration, analyzed = len(x)/RATE, (len(x)//HOP)*HOP/RATE
        noise = noise_estimates(x)  # validates common representation
        result = {"id": case["id"], "source_group": case["source_group"], "evidence_type": case["evidence_type"],
                  "duration_seconds": duration, "analyzed_duration_seconds": analyzed,
                  "ignored_incomplete_tail_seconds": duration-analyzed,
                  "analysis_samples_sha256": sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest(),
                  "noise_estimates_dbfs": noise}
        if case.get("sha256"):
            tokens = {case["source_group"], "audio:" + case["sha256"], "analysis-pcm:" + result["analysis_samples_sha256"],
                      *case.get("source_references", [])}
            if case.get("speaker_group"):
                tokens.add(case["speaker_group"])
            input_identities.append({"record_id": case["id"], "identity_tokens": sorted(tokens)})
        noise_ref = case.get("noise_reference_dbfs")
        if noise_ref is not None:
            if isinstance(noise_ref, bool) or not isinstance(noise_ref, (int, float)) or not np.isfinite(noise_ref):
                raise ValueError("Noise reference must be finite dBFS or null")
            result["noise_reference_dbfs"] = noise_ref
            result["noise_absolute_error_db"] = {k: abs(v-noise_ref) if v is not None else None for k, v in noise.items()}
        else:
            result["noise_absolute_error_db"] = None
        reference = case.get("level_events")
        if reference is not None:
            validate_intervals(reference, duration)
            result["level_changes"] = {}
            annotated_event_cases += 1
            if not reference:
                negative_seconds += duration
            for method in METHODS:
                predicted = level_changes(x, method)
                score = events(predicted, reference, duration)
                result["level_changes"][method] = {"predicted_intervals": predicted, "metrics": score}
                totals[method] += [score[k] for k in ("true_positive", "false_positive", "false_negative")]
                if not reference:
                    negative_fp[method] += score["false_positive"]
        else:
            result["level_changes"] = None
        speech = case.get("speech_intervals")
        vad = optional_webrtc(x)
        result["speech_methods"] = {"webrtc": vad}
        if silero_model is not None:
            result["speech_methods"]["silero_raw"] = silero_session.predict(x)
        if speech is not None:
            validate_intervals(speech, duration, nonoverlap=True)
            clipped = [{"start": item["start"], "end": min(analyzed, item["end"])} for item in speech if item["start"] < analyzed]
            result["speech_methods"]["energy_activity_diagnostic"] = speech_frames(energy_activity(x), clipped, analyzed)
            if vad["status"] == "available":
                vad["metrics"] = speech_frames(vad["intervals"], clipped, analyzed)
            if silero_model is not None:
                silero = result["speech_methods"]["silero_raw"]
                silero["metrics"] = speech_frames(silero["intervals"], clipped, analyzed)
        else:
            result["speech_methods"]["energy_activity_diagnostic"] = None
        results.append(result)
    identity_rows = [{"record_id": r["id"], "identity_tokens": ["source:" + r["source_group"], "pcm:" + r["analysis_samples_sha256"]]} for r in results]
    groups = connected_groups(identity_rows)
    evaluation_groups = {name: sha256("\n".join(group).encode()).hexdigest()[:16] for group in groups for name in group}
    paired = []
    for result in results:
        result["evaluation_group"] = evaluation_groups[result["id"]]
        if result["level_changes"] is not None:
            paired.append({"source_group": result["evaluation_group"],
                           **{method: result["level_changes"][method]["metrics"] for method in METHODS}})
    aggregate = {}
    for method in METHODS:
        metrics = count_metrics(*(int(v) for v in totals[method]))
        metrics["false_events_per_negative_hour"] = negative_fp[method]/(negative_seconds/3600) if negative_seconds else None
        aggregate[method] = metrics
    root = Path(__file__).resolve().parents[1]
    return {"schema_version": "method-benchmark-result-1.0", "config": CONFIG,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "code_sha256": {name: sha256((root/name).read_bytes()).hexdigest() for name in (
                "lab/research_analysis.py", "lab/research_metrics.py", "lab/research_benchmark.py", "lab/research_vad.py", "lab/signal_events.py")},
            "case_count": len(results), "source_groups": len(groups),
            "input_identities": input_identities,
            "annotated_event_cases": annotated_event_cases, "negative_event_seconds": negative_seconds,
            "event_aggregate": aggregate,
            "paired_event_f1_gain": paired_event_gain(paired, *METHODS, repetitions=repetitions),
            "results": results, "promotion_eligible": False,
            "promotion_blockers": ["Development-only runner; no locked independent evaluation performed.",
                                   "Natural-audio representativeness and annotation quality are not established by these results.",
                                   "Speech-positive support and calibrated noise/SNR accuracy require separate evidence."],
            "limitations": ["Event support intervals describe analysis windows, not verified edit boundaries.",
                            "No annotation is different from an explicitly empty annotation list.",
                            "Lower-tail noise estimates assume sufficiently quiet frames and can be biased by continuous speech or noise changes.",
                            "The Gaussian correction is invalid for correlated, colored, non-Gaussian or speech-filled noise frames unless separately validated.",
                            "Analytic fixture bootstrap bounds describe those constructed groups only; they do not support population performance claims."]}
