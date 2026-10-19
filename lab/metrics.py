"""Small, explicit metrics. Unavailable quantities are None, never zero."""
from collections import Counter
import math


def agreement(left, right):
    if len(left) != len(right):
        raise ValueError("Paired reviews must have equal length")
    n = len(left)
    if not n:
        return {"pairs": 0, "agreement": None, "kappa": None}
    observed = sum(a == b for a, b in zip(left, right)) / n
    a, b = Counter(left), Counter(right)
    expected = sum(a[k] * b[k] for k in a.keys() | b.keys()) / n**2
    return {"pairs": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def human_metrics(rows):
    rows = list(rows)
    decided = [r for r in rows if r["judgment"] != "uncertain"]
    return {"reviewed": len(rows), "decided": len(decided),
            "uncertainty_rate": sum(r["judgment"] == "uncertain" for r in rows) / len(rows) if rows else None,
            "accuracy_among_decided": sum(r["judgment"] == r["reference"] for r in decided) / len(decided) if decided else None}


def confusion(rows):
    rows = list(rows)
    genuine = [r for r in rows if r["reference"] == "human"]
    synthetic = [r for r in rows if r["reference"] == "synthetic"]
    rejected = sum(r["prediction"] == "synthetic" for r in genuine)
    accepted = sum(r["prediction"] == "human" for r in synthetic)
    return {"scored": len(rows), "genuine_total": len(genuine), "spoof_total": len(synthetic),
            "genuine_rejected": rejected, "spoof_accepted": accepted,
            "genuine_rejection_rate": rejected / len(genuine) if genuine else None,
            "spoof_acceptance_rate": accepted / len(synthetic) if synthetic else None,
            "genuine_accepted": len(genuine) - rejected, "spoof_rejected": len(synthetic) - accepted}


def equal_error_threshold(rows):
    """Tie-aware empirical operating point; ties select the lower threshold.

    Higher scores indicate bona fide speech; score >= threshold is human.
    This is a diagnostic threshold, not the detector's production threshold.
    """
    rows = list(rows)
    if any(not math.isfinite(r["score"]) for r in rows):
        raise ValueError("Scores must be finite")
    genuine = sum(r["reference"] == "human" for r in rows)
    spoof = len(rows) - genuine
    if not genuine or not spoof:
        raise ValueError("Threshold calibration requires both reference classes")
    grouped = {}
    for r in rows:
        counts = grouped.setdefault(r["score"], [0, 0])
        counts[r["reference"] != "human"] += 1
    scores = sorted(grouped)
    candidates = [math.nextafter(scores[0], -math.inf)]
    candidates += [(a / 2 + b / 2) for a, b in zip(scores, scores[1:])]
    candidates += [math.nextafter(scores[-1], math.inf)]
    rejected, accepted, cursor = 0, spoof, 0
    best = None
    for threshold in candidates:
        while cursor < len(scores) and scores[cursor] < threshold:
            g, s = grouped[scores[cursor]]
            rejected += g
            accepted -= s
            cursor += 1
        frr, far = rejected / genuine, accepted / spoof
        candidate = (abs(frr - far), threshold, (frr + far) / 2)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return {"threshold": best[1], "eer_at_operating_point": best[2],
            "n": len(rows), "genuine": genuine, "spoof": spoof,
            "direction": "higher_is_human", "comparison": "score >= threshold"}
