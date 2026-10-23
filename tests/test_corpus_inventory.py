import json
from hashlib import sha256

import pytest

from lab.corpus_inventory import MANIFESTS, connected_groups, inventory


def test_groups_connect_transitively_and_namespace_independent_sources():
    records = [{"record_id": "a", "identity_tokens": ["speaker:1", "ref:one"]},
               {"record_id": "b", "identity_tokens": ["ref:one", "audio:x"]},
               {"record_id": "c", "identity_tokens": ["audio:x"]},
               {"record_id": "d", "identity_tokens": ["other-dataset:speaker:1"]},
               {"record_id": "e", "identity_tokens": []}]
    assert connected_groups(records) == [["a", "b", "c"], ["d"], ["e"]]
    assert connected_groups(list(reversed(records))) == connected_groups(records)


def test_inventory_counts_shared_source_bytes_and_historical_speaker_overlap(tmp_path):
    base = tmp_path / "data/evaluation"
    base.mkdir(parents=True)
    audio = tmp_path / "data/example.wav"
    audio.write_bytes(b"fixture")
    cases = [{"id": "a", "path": "data/example.wav", "sha256": "a" * 64,
              "label": "human", "speaker": "one", "split": "selection", "source_utt": "u1"},
             {"id": "b", "path": "data/example.wav", "sha256": "b" * 64,
              "label": "synthetic", "speaker": "one", "split": "test", "source_utt": "u2"}]
    for name in MANIFESTS:
        (base / name).write_text(json.dumps({"cases": cases}))
    result = inventory(tmp_path)
    assert result["historical_case_rows"] == 8
    assert result["unique_original_byte_hashes"] == 2
    assert result["connected_identity_groups"] == 1
    assert len(result["groups_spanning_splits_within_a_manifest"]) == 4
    assert result["eligible_new_locked_groups"] == 0
    assert result["inventory_complete"]
    audio.unlink()
    result = inventory(tmp_path)
    assert not result["inventory_complete"]
    assert result["manifests"][0]["missing_files"] == ["data/example.wav"]


def test_human_vctk_mapping_supplies_speaker_identity(tmp_path):
    base = tmp_path / "data/evaluation"
    base.mkdir(parents=True)
    (tmp_path / "data/example.flac").write_bytes(b"fixture")
    case = {"id": "one", "trial_id": "LA_E_1", "path": "data/example.flac", "sha256": "a" * 64,
            "label": "human", "split": "test", "source_group": "vctk:p228_001"}
    for name in MANIFESTS:
        (base / name).write_text(json.dumps({"cases": [case]}))
    mapping = tmp_path / "data/asvspoof"
    mapping.mkdir()
    (mapping / "source_mapping.tsv").write_text(
        "ASVspoof_ID\tVCTK_ID\tTTS_VC_target_speaker\tVC_source_VCTK_ID\nLA_E_1\tp228_001\t-\t-\n")
    result = inventory(tmp_path)
    assert result["inventory_complete"]
    assert all(m["missing_speaker_identity"] == 0 for m in result["manifests"])


@pytest.mark.parametrize("schema,mode", [
    ("method-benchmark-result-1.0", "supplied development annotations"),
    ("core-detector-screen-1.0", "supplied development authenticity labels"),
    ("core-preprocessing-experiment-1.0", "supplied development authenticity labels"),
])
def test_method_comparisons_extend_prior_use_registry_and_tampering_is_rejected(tmp_path, schema, mode):
    base = tmp_path / "data/evaluation"
    base.mkdir(parents=True)
    for name in MANIFESTS:
        (base / name).write_text(json.dumps({"cases": []}))
    directory = base / "method-runs"
    directory.mkdir()
    receipt = {"schema_version": schema, "mode": mode,
               "input_identities": [{"record_id": "new-observed-recording", "identity_tokens": ["new:speaker:one", "audio:" + "a"*64]}]}
    text = json.dumps(receipt, indent=2)
    path = directory / (sha256(text.encode()).hexdigest() + ".json")
    path.write_text(text + "\n")
    result = inventory(tmp_path, include_records=True)
    assert len(result["identity_records"]) == 1
    assert "new:speaker:one" in result["identity_records"][0]["identity_tokens"]
    assert str(path.relative_to(tmp_path)) in result["method_run_file_sha256"]
    path.write_text(text.replace("new:speaker:one", "changed:speaker:one") + "\n")
    with pytest.raises(ValueError, match="Invalid prior-use comparison receipt"):
        inventory(tmp_path, include_records=True)
