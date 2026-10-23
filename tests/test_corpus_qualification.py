from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lab.corpus_qualification import qualify, qualify_file


@pytest.fixture
def prospective(tmp_path):
    contract = json.loads((Path(__file__).resolve().parents[1] / "docs/evaluation-contract-v1.json").read_text())
    contract["synthetic_speech"].update(critical_conditions=["noise"],
        minimum_independent_human_groups_each_critical_condition=1,
        minimum_independent_synthetic_groups_each_critical_condition=1)
    cases = []
    for i, partition in enumerate(("development", "calibration", "locked_test")):
        for j, label in enumerate(("human", "synthetic")):
            k = i*2+j
            path = tmp_path / f"case{k}.wav"
            audio = (.1*np.sin(2*np.pi*(300+100*k)*np.arange(16000)/16000)).astype(np.float32)
            sf.write(path, audio, 16000)
            cases.append({"id": str(k), "partition": partition, "label": label,
                "label_provenance": "controlled test metadata, not a real reference label", "path": path.name,
                "original_sha256": sha256(path.read_bytes()).hexdigest(), "duration_seconds": 1,
                "source_group": f"fixture:source:{k}", "speaker_group": f"fixture:speaker:{k}",
                "source_references": [], "generator_family": "bonafide" if label == "human" else ("new-family" if i == 2 else "dev-family"),
                "domain": "heldout-device" if i == 2 else "dev-device", "language": "en", "parent_id": None,
                "license": "test fixture only", "acquisition_source": "generated test fixture", "conditions": ["noise"]})
    path = tmp_path / "review.txt"
    path.write_text("Test-only audit file. This is not independent corpus evidence.")
    audits = {name: {"status": "passed", "reviewer": "test-only reviewer", "path": path.name,
                     "sha256": sha256(path.read_bytes()).hexdigest(), "conclusion": "documented_no_overlap"}
              for name in ("perceptual_duplicates", "label_provenance", "training_exposure", "previous_use")}
    document = {"schema_version": "prospective-corpus-1.0", "cases": cases, "audits": audits,
                "heldout_generator_families": ["new-family"], "heldout_domains": ["heldout-device"]}
    history = {"schema_version": "prior-use-registry-1.0", "records": [{"record_id": "old", "identity_tokens": ["old:speaker:one"]}]}
    return document, tmp_path, contract, history


def test_valid_structural_fixture_passes_without_claiming_performance(prospective):
    result = qualify(*prospective)
    assert result["structural_gates_passed"], result["errors"]
    assert result["connected_groups"] == 6
    assert "reviewer assertions" in result["scope"]


def test_transitive_source_reference_leakage_is_blocked(prospective):
    document, root, contract, history = prospective
    document["cases"][2]["source_references"] = [document["cases"][0]["source_group"]]
    document["cases"][4]["source_references"] = [document["cases"][2]["source_group"]]
    result = qualify(document, root, contract, history)
    assert not result["structural_gates_passed"]
    assert result["group_membership"]["0"] == result["group_membership"]["4"]
    assert any("crosses partitions" in e for e in result["errors"])


def test_history_matches_cloning_reference_and_missing_registry_blocks(prospective):
    document, root, contract, history = prospective
    document["cases"][4]["source_references"] = ["old:speaker:one"]
    result = qualify(document, root, contract, history)
    assert result["history_matches"][0]["id"] == "4"
    assert not result["structural_gates_passed"]
    assert any("prior-use registry" in e for e in qualify(document, root, contract)["errors"])


@pytest.mark.parametrize("change", ["missing_audit", "changed_audit", "training_unknown", "hash_mismatch", "missing_audio", "unknown_label"])
def test_unresolved_or_changed_evidence_prevents_qualification(prospective, change):
    document, root, contract, history = prospective
    if change == "missing_audit":
        del document["audits"]["previous_use"]
    elif change == "changed_audit":
        (root / "review.txt").write_text("changed")
    elif change == "training_unknown":
        document["audits"]["training_exposure"]["conclusion"] = "unknown"
    elif change == "hash_mismatch":
        document["cases"][0]["original_sha256"] = "f"*64
    elif change == "missing_audio":
        (root / "case0.wav").unlink()
    else:
        document["cases"][0]["label_provenance"] = "unknown"
    assert not qualify(document, root, contract, history)["structural_gates_passed"]


def test_duplicate_ids_parents_duration_and_heldout_contamination(prospective):
    document, root, contract, history = prospective
    broken = deepcopy(document)
    broken["cases"][1]["id"] = "0"
    with pytest.raises(ValueError, match="unique"):
        qualify(broken, root, contract, history)
    document["cases"][0]["parent_id"] = "1"
    document["cases"][1]["parent_id"] = "0"
    document["cases"][2]["duration_seconds"] = 2
    document["cases"][3]["generator_family"] = "new-family"
    errors = qualify(document, root, contract, history)["errors"]
    assert any("cyclic parent" in e for e in errors)
    assert any("duration differs" in e for e in errors)
    assert any("holdout" in e for e in errors)


def test_actual_contract_cannot_be_satisfied_by_tiny_fixture(prospective):
    document, root, _, history = prospective
    contract = json.loads((Path(__file__).resolve().parents[1] / "docs/evaluation-contract-v1.json").read_text())
    result = qualify(document, root, contract, history)
    assert not result["structural_gates_passed"]
    assert any("Insufficient independent support" in e for e in result["errors"])


def test_byte_identical_conflicting_labels_cannot_evade_graph(prospective):
    document, root, contract, history = prospective
    original = document["cases"][0]
    duplicate = document["cases"][5]
    duplicate["path"] = original["path"]
    duplicate["original_sha256"] = original["original_sha256"]
    errors = qualify(document, root, contract, history)["errors"]
    assert any("conflicting reference labels" in e for e in errors)
    assert any("crosses partitions" in e for e in errors)


def test_file_entry_point_requires_current_history_registry_evidence(prospective):
    document, root, contract, history = prospective
    (root / "docs").mkdir()
    (root / "docs/evaluation-contract-v1.json").write_text(json.dumps(contract))
    path = root / "prospective.json"
    path.write_text(json.dumps(document))
    history_path = root / "history.json"
    history_path.write_text(json.dumps(history))
    result = qualify_file(path, root, history_path)
    assert not result["structural_gates_passed"]
    assert any("missing/stale historical manifest evidence" in e for e in result["errors"])
    assert result["history_registry_sha256"] == sha256(history_path.read_bytes()).hexdigest()
