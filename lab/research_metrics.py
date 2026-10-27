"""Annotation-aware metrics with explicit undefined values and one-to-one events."""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def validate_intervals(intervals, duration, nonoverlap=False):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite")
    for item in intervals:
        if not all(isinstance(item[k], (int, float)) and not isinstance(item[k], bool)
                   and math.isfinite(item[k]) for k in ("start", "end")) or not 0 <= item["start"] < item["end"] <= duration:
            raise ValueError("Invalid annotation interval")
    ordered = sorted(intervals, key=lambda x: x["start"])
    if nonoverlap and any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Speech intervals must not overlap")


def count_metrics(tp, fp, fn):
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def events(predicted, reference, duration, minimum_iou=.5):
    validate_intervals(predicted, duration)
    validate_intervals(reference, duration)
    if not 0 < minimum_iou <= 1:
        raise ValueError("IoU must lie in (0, 1]")
    matches, errors = [], []
    if predicted and reference:
        iou = np.zeros((len(predicted), len(reference)))
        for i, p in enumerate(predicted):
            for j, r in enumerate(reference):
                intersection = max(0, min(p["end"], r["end"]) - max(p["start"], r["start"]))
                union = max(p["end"], r["end"]) - min(p["start"], r["start"])
                iou[i, j] = intersection / union
        valid = iou >= minimum_iou
        # Maximize cardinality first, total valid IoU second. Invalid edges
        # contribute zero, so a strong match cannot displace two valid matches.
        weight = np.where(valid, min(iou.shape) + 1 + iou, 0)
        left, right = linear_sum_assignment(weight, maximize=True)
        for i, j in zip(left, right):
            if valid[i, j]:
                matches.append({"prediction": int(i), "reference": int(j), "iou": float(iou[i, j])})
                errors.extend([abs(predicted[i]["start"] - reference[j]["start"]),
                               abs(predicted[i]["end"] - reference[j]["end"])])
    result = count_metrics(len(matches), len(predicted)-len(matches), len(reference)-len(matches))
    result.update(matches=matches, minimum_iou=minimum_iou,
                  matched_boundary_error_median_seconds=float(np.median(errors)) if errors else None,
                  matched_boundary_error_p95_seconds=float(np.quantile(errors, .95)) if errors else None,
                  boundary_scope="matched events only; unmatched events counted separately")
    return result


def speech_frames(predicted, reference, analyzed_duration, hop=.02):
    validate_intervals(predicted, analyzed_duration, nonoverlap=True)
    validate_intervals(reference, analyzed_duration, nonoverlap=True)
    if not math.isfinite(hop) or hop <= 0:
        raise ValueError("Frame hop must be positive and finite")
    count = round(analyzed_duration/hop)
    if count < 1 or abs(count*hop-analyzed_duration) > 1e-8:
        raise ValueError("Evaluation duration must be a complete number of frames")
    centers = (np.arange(count) + .5)*hop
    def mask(intervals):
        result = np.zeros(count, dtype=bool)
        for interval in intervals:
            result |= (centers >= interval["start"]) & (centers < interval["end"])
        return result
    p, r = mask(predicted), mask(reference)
    result = count_metrics(int(np.sum(p & r)), int(np.sum(p & ~r)), int(np.sum(~p & r)))
    negative = int(np.sum(~r))
    result.update(frames=count, hop_seconds=hop, true_negative=int(np.sum(~p & ~r)),
                  nonspeech_false_positive_rate=float(np.sum(p & ~r)/negative) if negative else None,
                  boundary_rule="frame center in [start, end); incomplete final audio frame excluded")
    return result


def paired_event_gain(rows, baseline, candidate, repetitions=10000, seed=20261022):
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap draws required")
    groups = {}
    for row in rows:
        if not row.get("source_group"):
            raise ValueError("Source grouping required")
        values = groups.setdefault(row["source_group"], np.zeros((2, 3)))
        for i, method in enumerate((baseline, candidate)):
            values[i] += [row[method][k] for k in ("true_positive", "false_positive", "false_negative")]
    if len(groups) < 2:
        return {"groups": len(groups), "gain_percentile95": None, "valid_draws": 0}
    counts = np.array([groups[k] for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    # Process draws in small blocks: group count need not determine peak memory.
    gains = []
    for start in range(0, repetitions, 100):
        n = min(100, repetitions-start)
        draws = counts[rng.integers(len(counts), size=(n, len(counts)))].sum(axis=1)
        denominator = 2*draws[:, :, 0] + draws[:, :, 1] + draws[:, :, 2]
        valid = np.all(denominator > 0, axis=1)
        f1 = 2*draws[valid, :, 0]/denominator[valid]
        gains.extend((f1[:, 1]-f1[:, 0]).tolist())
    return {"groups": len(groups), "repetitions": repetitions, "seed": seed,
            "valid_draws": len(gains), "gain_percentile95": np.quantile(gains, [.025, .975]).tolist() if gains else None,
            "method": "paired source-group percentile bootstrap of micro event F1 difference"}
