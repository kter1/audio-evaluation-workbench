"""Offline baseline audit. Archived score replay is explicitly not fresh inference."""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import numpy as np

from lab.evaluation import summarize, wilson
from lab.store import digest
from lab.fusion import sigmoid


def overlap(left, right):
    """Compare development and evaluation, including cloning-reference utterances."""
    def values(cases, fields):
        return {str(c[field]) for c in cases for field in fields if c.get(field)}

    fields = {
        "speaker": ("speaker",), "audio_sha256": ("sha256",),
        "source_or_reference_utterance": ("source_utt", "ref_utt"),
    }
    result = {}
    for name, keys in fields.items():
        common = values(left, keys) & values(right, keys)
        result[name] = {"count": len(common), "examples": sorted(common)[:10]}
    texts = lambda cases: {" ".join(c["text"].casefold().split()) for c in cases if c.get("text")}
    common = texts(left) & texts(right)
    result["normalized_text"] = {"count": len(common), "examples": sorted(common)[:3]}
    return result


def cluster_intervals(rows, threshold, repetitions=2000, seed=20261022):
    """Percentile bootstrap of speakers, retaining all recordings/variants per draw.

    These intervals are descriptive, not certification of rare-error performance.
    Missing class draws are omitted and their valid counts reported.
    """
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    groups = defaultdict(lambda: np.zeros(4, dtype=float))
    for row in rows:
        if not row.get("speaker"):
            raise ValueError("Speaker identity required for clustered uncertainty")
        human = row["label"] == "human"
        flagged = row["score"] >= threshold
        groups[str(row["speaker"])] += [human, human and flagged, not human, not human and flagged]
    if len(groups) < 2:
        return {"method": "speaker percentile bootstrap", "speakers": len(groups),
                "false_positive_rate": None, "detection_rate": None,
                "reason": "At least two speaker clusters required"}
    counts = np.array([groups[k] for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    draws = counts[rng.integers(len(counts), size=(repetitions, len(counts)))].sum(axis=1)
    result = {"method": "speaker percentile bootstrap", "speakers": len(groups),
              "repetitions": repetitions, "seed": seed,
              "caution": "Zero observed errors yield degenerate bootstrap bounds; not evidence of zero population risk."}
    for key, num, den in (("false_positive_rate", 1, 0), ("detection_rate", 3, 2)):
        valid = draws[:, den] > 0
        rates = draws[valid, num] / draws[valid, den]
        result[key] = {"percentile95": np.quantile(rates, [.025, .975]).tolist() if len(rates) else None,
                       "valid_draws": int(valid.sum())}
    return result


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside repository: {relative}")
    return path


def validate_rows(rows, cases, channels):
    expected = {(c["id"], ch) for c in cases for ch in channels}
    indexed = {c["id"]: c for c in cases}
    if len(indexed) != len(cases):
        raise ValueError("Duplicate manifest case IDs")
    seen = set()
    for row in rows:
        key = (row["id"], row["channel"])
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected receipt row: {key}")
        seen.add(key)
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError(f"Invalid score: {key}")
        case = indexed[row["id"]]
        for field in ("label", "speaker", "generator", "source"):
            if row[field] != case[field]:
                raise ValueError(f"Receipt/manifest {field} mismatch: {key}")
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} receipt rows")


def audit(root, repetitions=2000):
    root = Path(root).resolve()
    paths = {
        "receipt": "docs/detector-evaluation.json", "fusion": "lab/fusion-v4.json",
        "development": "data/evaluation/modern-manifest.json",
        "evaluation": "data/evaluation/v4-manifest.json",
        "development_features": "data/v4-dev/features.json",
    }
    documents = {k: json.loads((root / p).read_text()) for k, p in paths.items()}
    receipt, spec = documents["receipt"], documents["fusion"]
    dev, test = documents["development"], documents["evaluation"]
    errors = []
    hashes = {k: sha256((root / p).read_bytes()).hexdigest() for k, p in paths.items()}
    for name, manifest in (("development", dev), ("evaluation", test)):
        if digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
            errors.append(f"{name} manifest digest mismatch")
    if receipt["manifest_sha256"] != test["sha256"]:
        errors.append("Receipt evaluation manifest mismatch")
    if spec["development"]["v3_manifest_sha256"] != dev["sha256"]:
        errors.append("Fusion development manifest mismatch")
    if receipt["fusion_sha256"] != hashes["fusion"] or receipt["models"] != spec["models"]:
        errors.append("Receipt fusion/model identity mismatch")
    if receipt["threshold"] != spec["threshold"]:
        errors.append("Receipt fusion threshold mismatch")
    features = documents["development_features"]
    if hashes["development_features"] != spec["development"]["features_sha256"]:
        errors.append("Development features hash mismatch")
    if features["models"] != spec["models"] or features["v3_manifest_sha256"] != dev["sha256"]:
        errors.append("Development feature model/manifest identity mismatch")
    feature_keys = {(r["id"], r["channel"]) for r in features["rows"].values()}
    expected_features = {(c["id"], ch) for c in dev["cases"] for ch in features["channels"]}
    if feature_keys != expected_features or len(feature_keys) != len(features["rows"]):
        errors.append("Development feature coverage mismatch")
    rows, threshold, channels = receipt["test"]["rows"], receipt["threshold"], test["channels"]
    validate_rows(rows, test["cases"], channels)
    leakage = overlap(dev["cases"], test["cases"])
    for name in ("speaker", "audio_sha256", "source_or_reference_utterance"):
        if leakage[name]["count"]:
            errors.append(f"Development/evaluation overlap: {name}")
    heldout = test["heldout_generator"]
    if any(c["generator"] == heldout for c in dev["cases"]):
        errors.append("Held-out generator appears in development")

    # Verify every local source/derivative in both manifests, streaming large files.
    audio = {"checked": 0, "missing": [], "mismatched": []}
    verified = set()
    for manifest in (dev, test):
        for case in manifest["cases"]:
            variants = [case, *case.get("channels", {}).values()] if isinstance(case.get("channels"), dict) else [case]
            for variant in variants:
                key = (variant["path"], variant["sha256"])
                if key in verified:
                    continue
                verified.add(key)
                path = checked_path(root, variant["path"])
                if not path.is_file():
                    audio["missing"].append(variant["path"])
                else:
                    with path.open("rb") as stream:
                        from hashlib import file_digest
                        actual = file_digest(stream, "sha256").hexdigest()
                    audio["checked"] += 1
                    if actual != variant["sha256"]:
                        audio["mismatched"].append(variant["path"])
    if audio["missing"] or audio["mismatched"]:
        errors.append("Local audio missing or hash mismatch")

    # Content-addressed historical reports: replay window aggregation and compare receipt.
    reports = defaultdict(list)
    for path in sorted((root / "data/evaluation/v4-runs").glob("*.json")):
        raw = path.read_bytes()
        report = json.loads(raw)
        if sha256(raw.rstrip(b"\n")).hexdigest() != path.stem:
            errors.append(f"Archived report content hash mismatch: {path.name}")
        reports[report["source"]["sha256"]].append(report)
    replayed, worst, runtimes, durations = 0, 0.0, [], []
    cases = {c["id"]: c for c in test["cases"]}
    for row in rows:
        case = cases[row["id"]]
        variant = case if row["channel"] == "clean" else case["channels"][row["channel"]]
        candidates = [r for r in reports[variant["sha256"]]
                      if r["model"].get("fusion_sha256") == hashes["fusion"]]
        if len(candidates) != 1:
            errors.append(f"Expected one matching archived report: {row['id']}/{row['channel']}")
            continue
        report = candidates[0]
        weights = [w["aggregation_weight"] for w in report["windows"]]
        if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Invalid archived aggregation weights")
        if report["policy"]["binary_threshold"] != threshold or report["model"]["components"] != spec["models"]:
            errors.append(f"Archived policy/model mismatch: {row['id']}/{row['channel']}")
        for window in report["windows"]:
            if not math.isfinite(window["synthetic_score"]) or not 0 <= window["synthetic_score"] <= 1:
                raise ValueError("Invalid archived window score")
            if abs(float(sigmoid(window["logits"][1] - window["logits"][0])) - window["synthetic_score"]) > 1e-10:
                errors.append(f"Archived logit/score mismatch: {row['id']}/{row['channel']}")
        score = float(np.average([w["synthetic_score"] for w in report["windows"]], weights=weights))
        delta = max(abs(score - row["score"]), abs(score - report["assessment"]["synthetic_score"]))
        worst = max(worst, delta)
        if delta > 1e-10 or len(report["windows"]) != row["windows"]:
            errors.append(f"Archived aggregation mismatch: {row['id']}/{row['channel']}")
        if report["assessment"]["assessment"] != row["assessment"]:
            errors.append(f"Archived assessment mismatch: {row['id']}/{row['channel']}")
        replayed += 1
        runtimes.append(report["runtime_seconds"])
        durations.append(report["source"]["duration_seconds"])

    results = {}
    for channel in channels:
        part = [r for r in rows if r["channel"] == channel]
        summary = summarize(part, threshold)
        for label, key in (("human", "reals"), ("synthetic", "fakes")):
            subset = [r for r in part if r["label"] == label]
            flagged = sum(r["score"] >= threshold for r in subset)
            expected = receipt["test"]["by_channel"][channel][key]
            if flagged != expected["flagged_synthetic"] or len(subset) != expected["n"]:
                errors.append(f"Published counts mismatch: {channel}/{label}")
        results[channel] = {"summary": summary, "speaker_uncertainty": cluster_intervals(part, threshold, repetitions),
                            "false_positive_wilson95_descriptive": wilson(summary["false_positive"], summary["human"])}
    return {
        "schema_version": "baseline-audit-1.0", "mode": "archived replay; no fresh inference",
        "input_sha256": hashes, "errors": errors, "integrity_passed": not errors,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "audit_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "development_evaluation_overlap": leakage, "local_audio": audio,
        "evaluation": {"source_recordings": len(test["cases"]), "clip_versions": len(rows),
                       "speakers": len({c["speaker"] for c in test["cases"]}),
                       "heldout_generator": heldout, "channels": results},
        "archived_replay": {"rows": replayed, "maximum_score_difference": worst,
                            "historical_runtime_seconds_median": float(np.median(runtimes)) if runtimes else None,
                            "historical_runtime_seconds_p95": float(np.quantile(runtimes, .95)) if runtimes else None,
                            "historical_total_runtime_over_audio_duration": sum(runtimes) / sum(durations) if durations else None,
                            "peak_memory": None},
        "development_target_failures": spec["cross_validation"]["fusion"]["failures"],
        "eligible_as_locked_test": False,
        "limitations": [
            "Published evaluation has been consulted; it is regression evidence, not an untouched test.",
            "No check proves absence of upstream model-training contamination.",
            "Byte hashes do not detect perceptual duplicates; normalized text comparison is whitespace/case only.",
            "Speaker bootstrap captures observed speaker clustering, not unseen domains or generator uncertainty.",
            "Historical runtime is environment-specific; peak memory was not recorded.",
            "This command audits v4 fusion development versus evaluation, not every historical dataset.",
            "No manipulation-localization, multilingual, conversational, or live replay validation is established.",
        ],
    }
