import json
import importlib.metadata
from hashlib import sha256
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lab.research_analysis import energy_activity, level_changes, noise_estimates, optional_webrtc
from lab.research_benchmark import load_manifest, run
from lab.research_metrics import events, paired_event_gain, speech_frames, validate_intervals


def test_event_matching_is_one_to_one_and_maximizes_match_count():
    reference = [{"start": 0, "end": 1}, {"start": 1, "end": 2}]
    predicted = [{"start": 0, "end": 2}, {"start": 0, "end": 1}]
    result = events(predicted, reference, 3)
    assert result["true_positive"] == 2
    assert result["false_negative"] == 0
    duplicate = events([reference[0], reference[0]], [reference[0]], 3)
    assert duplicate["true_positive"] == 1 and duplicate["false_positive"] == 1
    assert duplicate["matched_boundary_error_p95_seconds"] == 0


def test_no_annotations_does_not_become_perfect_precision_or_recall():
    result = events([], [], 1)
    assert result["precision"] is None and result["recall"] is None and result["f1"] is None
    assert result["matched_boundary_error_p95_seconds"] is None
    result = speech_frames([], [], 1)
    assert result["recall"] is None and result["nonspeech_false_positive_rate"] == 0


@pytest.mark.parametrize("interval", [{"start": -1, "end": 1}, {"start": 0, "end": 2},
    {"start": 1, "end": 0}, {"start": 0, "end": float("nan")}, {"start": "0", "end": 1}])
def test_bad_annotations_are_rejected(interval):
    with pytest.raises(ValueError):
        validate_intervals([interval], 1)


def test_frame_centers_define_boundaries_and_missing_class_is_explicit():
    result = speech_frames([{"start": 0, "end": .04}], [{"start": .02, "end": .06}], .1)
    assert (result["true_positive"], result["false_positive"], result["false_negative"], result["true_negative"]) == (1, 1, 1, 2)
    with pytest.raises(ValueError, match="overlap"):
        speech_frames([], [{"start": 0, "end": .06}, {"start": .04, "end": .08}], .1)


def test_energy_is_not_speech_and_optional_vad_cannot_invent_scores(monkeypatch):
    noise = np.random.default_rng(1).normal(0, .02, 16000).astype(np.float32)
    result = speech_frames(energy_activity(noise), [], 1)
    assert result["nonspeech_false_positive_rate"] == 1
    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    unavailable = optional_webrtc(noise)
    assert unavailable["status"] == "unavailable"
    assert "intervals" not in unavailable


def test_adaptive_rule_detects_a_known_large_gain_step_and_ignores_flat_tone():
    x = (.1*np.sin(2*np.pi*1000*np.arange(6*16000)/16000)).astype(np.float32)
    assert level_changes(x, "adaptive_level_rule") == []
    x[3*16000:] *= 8
    for method in ("frozen_level_rule", "adaptive_level_rule"):
        predicted = level_changes(x, method)
        result = events(predicted, [{"start": 2.88, "end": 3.12}], 6)
        assert result["true_positive"] == 1
        assert result["matched_boundary_error_p95_seconds"] <= .02


def test_noise_statistic_is_unavailable_in_db_for_silence():
    assert all(v is None for v in noise_estimates(np.zeros(16000)).values())
    constant = noise_estimates(np.full(16000, .1))
    assert constant["global_lower_decile"] == pytest.approx(-20)
    assert constant["blockwise_lower_decile_median"] == pytest.approx(-20)
    # A constant is not Gaussian noise, so the assumption-specific correction
    # must not be represented as an exact estimator for this confounder.
    assert constant["iid_gaussian_corrected_decile"] > -20


def test_paired_bootstrap_retains_source_group_and_is_repeatable():
    rows = [{"source_group": "a", "base": {"true_positive": 0, "false_positive": 0, "false_negative": 1},
             "candidate": {"true_positive": 1, "false_positive": 0, "false_negative": 0}},
            {"source_group": "b", "base": {"true_positive": 1, "false_positive": 0, "false_negative": 0},
             "candidate": {"true_positive": 1, "false_positive": 0, "false_negative": 0}}]
    result = paired_event_gain(rows, "base", "candidate", 500)
    assert result == paired_event_gain(rows + rows, "base", "candidate", 500)
    assert result["groups"] == 2 and result["valid_draws"] == 500
    assert result["gain_percentile95"][0] >= 0


def test_development_runner_refuses_locked_labels_before_loading_files(tmp_path):
    path = tmp_path / "locked.json"
    path.write_text(json.dumps({"schema_version": "method-benchmark-1.0", "partition": "locked_test", "cases": []}))
    with pytest.raises(ValueError, match="development"):
        load_manifest(path, tmp_path)


def test_benchmark_distinguishes_unknown_annotations_from_verified_empty_and_never_promotes(monkeypatch):
    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    common = {"source_group": "fixture:a", "samples": np.zeros(16001), "evidence_type": "analytic_fixture"}
    result = run([{**common, "id": "unknown"}, {**common, "id": "empty", "level_events": [], "speech_intervals": []}], 100)
    assert result["results"][0]["level_changes"] is None
    assert result["results"][0]["speech_methods"]["energy_activity_diagnostic"] is None
    assert result["results"][1]["level_changes"]["adaptive_level_rule"]["metrics"]["false_positive"] == 0
    assert result["results"][1]["ignored_incomplete_tail_seconds"] == pytest.approx(1/16000)
    assert result["annotated_event_cases"] == 1
    assert not result["promotion_eligible"]
    json.dumps(result, allow_nan=False)


def test_identical_decoded_audio_cannot_inflate_bootstrap_group_count(monkeypatch):
    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    common = {"samples": np.zeros(16000), "evidence_type": "analytic_fixture", "level_events": []}
    result = run([{**common, "id": "a", "source_group": "claimed:a"},
                  {**common, "id": "b", "source_group": "claimed:b"}], 100)
    assert result["source_groups"] == 1
    assert result["results"][0]["evaluation_group"] == result["results"][1]["evaluation_group"]
    assert result["paired_event_f1_gain"]["gain_percentile95"] is None


def test_gaussian_correction_reduces_known_finite_frame_bias_on_white_noise():
    # Mechanism check under the declared distribution, not validation on natural noise.
    noise = np.random.default_rng(123).normal(0, .02, 30*16000).astype(np.float32)
    reference = 20*np.log10(np.sqrt(np.mean(noise.astype(float)**2)))
    estimates = noise_estimates(noise)
    assert abs(estimates["iid_gaussian_corrected_decile"]-reference) < .1
    assert estimates["global_lower_decile"] < reference-.2


def test_optional_vad_adapter_has_explicit_pcm_conversion_and_wheel_provenance(monkeypatch):
    calls = []
    class FakeVad:
        def __init__(self, mode):
            assert mode == 2
        def is_speech(self, frame, rate):
            calls.append((frame, rate))
            return False  # Mechanical adapter fixture, not a model prediction.
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=FakeVad))
    def version(name):
        if name == "webrtcvad":
            raise importlib.metadata.PackageNotFoundError(name)
        return "test-only"
    monkeypatch.setattr(importlib.metadata, "version", version)
    x = np.tile([1.2, -1.2], 8001)
    result = optional_webrtc(x)
    assert result["distribution"] == "webrtcvad-wheels"
    assert result["package_version"] == "test-only"
    assert len(calls) == 50 and all(len(frame) == 640 and rate == 16000 for frame, rate in calls)
    assert np.array_equal(np.frombuffer(calls[0][0], dtype="<i2")[:2], [32767, -32768])
    assert result["clipped_input_samples"] == len(x)


def test_real_file_development_runs_issue_durable_idempotent_prior_use_receipts(tmp_path, monkeypatch):
    from scripts import benchmark_methods
    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    audio = tmp_path / "recording.wav"
    sf.write(audio, np.zeros(16000, np.float32), 16000)
    path = tmp_path / "development.json"
    path.write_text(json.dumps({"schema_version": "method-benchmark-1.0", "partition": "development", "cases": [{
        "id": "observed", "source_group": "fixture:speaker:one", "path": audio.name,
        "sha256": sha256(audio.read_bytes()).hexdigest(), "annotation_provenance": "Test only; no annotations"}]}))
    monkeypatch.setattr(benchmark_methods, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark_methods.py", "--manifest", str(path), "--bootstrap", "100"])
    assert benchmark_methods.main() == 0
    receipts = list((tmp_path / "data/evaluation/method-runs").glob("*.json"))
    assert len(receipts) == 1
    content = receipts[0].read_bytes()
    result = json.loads(content)
    assert "fixture:speaker:one" in result["input_identities"][0]["identity_tokens"]
    assert "audio:" + sha256(audio.read_bytes()).hexdigest() in result["input_identities"][0]["identity_tokens"]
    assert benchmark_methods.main() == 0
    assert receipts[0].read_bytes() == content
    assert len(list(receipts[0].parent.glob("*.json"))) == 1
