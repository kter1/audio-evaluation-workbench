"""Integrity failures must be distinguishable from disappointing model performance."""
from hashlib import sha256
import json
import math

import pytest

from lab.baseline_audit import audit, checked_path, cluster_intervals, overlap, validate_rows
from lab.store import digest


def test_overlap_includes_cloning_references_and_normalized_text():
    dev = [{"speaker": "a", "source_utt": "u1", "ref_utt": "r1", "sha256": "x", "text": "Hello  WORLD"}]
    test = [{"speaker": "b", "source_utt": "r1", "sha256": "y", "text": "hello world"}]
    result = overlap(dev, test)
    assert result["speaker"]["count"] == 0
    assert result["source_or_reference_utterance"]["count"] == 1
    assert result["normalized_text"]["count"] == 1


def test_bootstrap_keeps_correlated_variants_together_and_is_repeatable():
    rows = [{"speaker": str(i), "label": "human" if i < 4 else "synthetic",
             "score": .8 if i in (0, 4, 5, 6) else .1} for i in range(8)]
    one = cluster_intervals(rows, .5, 500)
    assert one == cluster_intervals(rows, .5, 500)
    doubled = cluster_intervals(rows + rows, .5, 500)
    assert one == doubled
    bounds = one["false_positive_rate"]["percentile95"]
    assert bounds[0] <= .25 <= bounds[1]
    with pytest.raises(ValueError, match="Speaker"):
        cluster_intervals([{"label": "human", "score": .2}], .5)


def test_missing_classes_and_insufficient_clusters_are_explicit():
    assert cluster_intervals([{"speaker": "a", "label": "human", "score": 0}], .5)["false_positive_rate"] is None
    rows = [{"speaker": s, "label": "human", "score": 0} for s in ("a", "b")]
    result = cluster_intervals(rows, .5, 100)
    assert result["detection_rate"]["valid_draws"] == 0
    assert result["detection_rate"]["percentile95"] is None
    assert "Zero observed" in result["caution"]


def test_receipt_coverage_metadata_and_scores_fail_closed():
    cases = [{"id": "one", "label": "human", "speaker": "a", "generator": "bonafide", "source": "x"}]
    row = {**cases[0], "channel": "clean", "score": .2}
    validate_rows([row], cases, ["clean"])
    for bad in ([], [row, row], [{**row, "label": "synthetic"}], [{**row, "score": float("nan")}],
                [{**row, "score": 2}]):
        with pytest.raises(ValueError):
            validate_rows(bad, cases, ["clean"])


def test_manifest_paths_cannot_escape_repository(tmp_path):
    assert checked_path(tmp_path, "audio.wav") == tmp_path / "audio.wav"
    with pytest.raises(ValueError, match="outside"):
        checked_path(tmp_path, "../outside.wav")


@pytest.fixture
def baseline(tmp_path):
    def write(path, data):
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        return sha256(dest.read_bytes()).hexdigest()

    cases = []
    for i, label in enumerate(("human", "synthetic")):
        raw = f"audio{i}".encode()
        path = f"data/audio{i}"
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / path).write_bytes(raw)
        cases.append({"id": str(i), "label": label, "speaker": str(i), "source": "fixture",
                      "generator": "bonafide" if i == 0 else "heldout", "path": path,
                      "sha256": sha256(raw).hexdigest(), "source_utt": f"utt{i}"})
    dev = {"cases": []}
    dev["sha256"] = digest(dev)
    test = {"cases": cases, "channels": ["clean"], "heldout_generator": "heldout"}
    test["sha256"] = digest(test)
    write("data/evaluation/modern-manifest.json", dev)
    write("data/evaluation/v4-manifest.json", test)
    features = {"rows": {}, "channels": ["clean"], "models": {}, "v3_manifest_sha256": dev["sha256"]}
    feature_hash = write("data/v4-dev/features.json", features)
    spec = {"threshold": .5, "models": {},
            "development": {"v3_manifest_sha256": dev["sha256"], "features_sha256": feature_hash},
            "cross_validation": {"fusion": {"failures": []}}}
    fusion_hash = write("lab/fusion-v4.json", spec)
    rows = []
    for i, case in enumerate(cases):
        score = .1 if i == 0 else .9
        row = {k: case[k] for k in ("id", "label", "speaker", "generator", "source")}
        row.update(channel="clean", score=score, assessment="fixture", windows=1)
        rows.append(row)
        report = {"source": {"sha256": case["sha256"], "duration_seconds": 4},
                  "model": {"fusion_sha256": fusion_hash, "components": {}},
                  "policy": {"binary_threshold": .5},
                  "windows": [{"aggregation_weight": 1, "synthetic_score": score,
                               "logits": [0, math.log(score / (1 - score))]}],
                  "assessment": {"synthetic_score": score, "assessment": "fixture"}, "runtime_seconds": 1}
        text = json.dumps(report, indent=2, allow_nan=False)
        write(f"data/evaluation/v4-runs/{sha256(text.encode()).hexdigest()}.json", report)
    receipt = {"threshold": .5, "models": {}, "fusion_sha256": fusion_hash, "manifest_sha256": test["sha256"],
               "test": {"rows": rows, "by_channel": {"clean": {
                   "reals": {"n": 1, "flagged_synthetic": 0}, "fakes": {"n": 1, "flagged_synthetic": 1}}}}}
    write("docs/detector-evaluation.json", receipt)
    return tmp_path


def test_end_to_end_archive_replay_and_audio_tamper_detection(baseline):
    result = audit(baseline, 100)
    assert result["integrity_passed"], result["errors"]
    assert result["archived_replay"]["rows"] == 2
    assert not result["eligible_as_locked_test"]
    (baseline / "data/audio0").write_bytes(b"changed")
    result = audit(baseline, 100)
    assert not result["integrity_passed"]
    assert result["local_audio"]["mismatched"] == ["data/audio0"]


def test_tampered_receipt_counts_are_detected(baseline):
    path = baseline / "docs/detector-evaluation.json"
    data = json.loads(path.read_text())
    data["test"]["by_channel"]["clean"]["reals"]["flagged_synthetic"] = 1
    path.write_text(json.dumps(data))
    result = audit(baseline, 100)
    assert "Published counts mismatch: clean/human" in result["errors"]
