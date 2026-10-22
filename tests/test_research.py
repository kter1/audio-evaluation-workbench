import json
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pytest

from lab.research import ResearchDB, validate_manifest, _inside_media_root, safe_csv

DEMO = Path(__file__).resolve().parents[1] / "fixtures/demo_research.json"
GOLD = Path(__file__).resolve().parents[1] / "fixtures/expected_findings.json"

@pytest.fixture
def db(tmp_path):
    instance = ResearchDB(tmp_path / "research.sqlite")
    yield instance
    instance.close()

@pytest.fixture
def demo():
    return json.loads(DEMO.read_text())

def test_manifest_immutability_and_idempotency(db, demo):
    assert db.import_manifest(demo) is True
    assert db.import_manifest(demo) is False
    changed = {**demo, "name": "Different"}
    with pytest.raises(ValueError, match="different content"):
        db.import_manifest(changed)
    assert len(db.datasets()) == 1

def test_schema_rollback_and_bad_input(db, demo):
    broken = {**demo, "recordings": [{"recording_id": "ok"}, "not an object"]}
    with pytest.raises(ValueError, match="object"):
        db.import_manifest(broken)
    assert db.datasets() == []
    for value in [float("nan"), float("inf")]:
        bad = {**demo, "recordings": [{"recording_id": "x", "duration_seconds": value}]}
        with pytest.raises(ValueError):
            validate_manifest(bad)

def test_expected_flags_are_detected_and_repeat_runs_keep_state(db, demo, tmp_path):
    db.import_manifest(demo)
    first = db.audit(demo["dataset_id"], media_root=tmp_path)
    flags = db.issues(demo["dataset_id"])
    target = json.loads(GOLD.read_text())
    assert set(target["rules"]) == {r["rule_id"] for r in flags}
    assert first >= target["minimum_flags"]
    assert all(r["id"] and r["next_check"] and r["evidence"] for r in flags)
    ids = {r["id"] for r in flags}
    assert db.audit(demo["dataset_id"], media_root=tmp_path) == first
    assert ids == {r["id"] for r in db.issues(demo["dataset_id"])}

def test_issue_transitions_are_append_only(db, demo, tmp_path):
    db.import_manifest(demo)
    db.audit(demo["dataset_id"], media_root=tmp_path)
    issue = db.issues(demo["dataset_id"])[0]["id"]
    db.disposition(issue, "investigating", "demo-analyst", "Checking source metadata")
    db.disposition(issue, "resolved", "demo-analyst", "Corrected in a separately versioned source")
    record = db.issue(issue)
    assert record["status"] == "resolved"
    assert [r["status"] for r in record["events"]] == ["investigating", "resolved"]
    assert len(db.issues(demo["dataset_id"], status="resolved")) == 1
    with pytest.raises(ValueError, match="required"):
        db.disposition(issue, "dismissed", "analyst", "")

def test_no_media_traversal_or_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"w")
    (root / "alias.wav").symlink_to(outside)
    assert _inside_media_root(root, "../outside.wav") is None
    assert _inside_media_root(root, str(outside)) is None
    assert _inside_media_root(root, "alias.wav") is None
    assert _inside_media_root(root, "normal.wav") == root / "normal.wav"

def test_detect_actual_hash_mismatch(db, demo, tmp_path):
    p = tmp_path / "sound.wav"
    p.write_bytes(b"not a real audio file")
    demo["dataset_id"] = "tampered"
    demo["recordings"][0]["media_path"] = "sound.wav"
    demo["recordings"][0]["sha256"] = "0" * 64
    db.import_manifest(demo)
    db.audit("tampered", tmp_path)
    assert "HASH_MISMATCH" in {i["rule_id"] for i in db.issues("tampered")}

def test_exports_escape_html_and_spreadsheet_formula(db, demo, tmp_path):
    demo["dataset_id"] = "escape-case"
    demo["name"] = "<img src=x onerror=alert(1)>"
    demo["annotations"][0]["source"] = "=2+2"
    db.import_manifest(demo)
    db.audit("escape-case", tmp_path)
    pack = db.export_bundle("escape-case")
    with ZipFile(BytesIO(pack)) as z:
        files = set(z.namelist())
        assert {"summary.json", "issues.csv", "disposition_history.csv", "audit.html"} <= files
        html = z.read("audit.html").decode()
        assert "<img" not in html
        assert "&lt;img" in html
        summary = json.loads(z.read("summary.json"))
        assert summary["recordings"] == len(demo["recordings"])
    rows = safe_csv([{"value": "=HYPERLINK('bad')"}], ["value"]).decode("utf-8-sig")
    assert "'=HYPERLINK" in rows

def test_missing_reference_not_model_proof(db, demo, tmp_path):
    db.import_manifest(demo)
    db.audit(demo["dataset_id"], tmp_path)
    review = [r for r in db.issues(demo["dataset_id"]) if r["rule_id"] == "CONFLICTING_LABELS"]
    assert len(review) == 1
    assert review[0]["severity"] == "review"
    assert "model" not in review[0]["next_check"].lower()
