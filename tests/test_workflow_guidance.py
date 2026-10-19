from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import zipfile

from conftest import submit_phase, payload
from lab.reporting import calibration_feedback, findings, research_bundle
from lab.handoff import export_package, package_bytes
from lab.workflow import progress


def test_guidance_goes_from_listening_to_calibration_feedback(db):
    state = progress(db, "practice-reviewer")
    assert state["next"]["screen"] == "Review"
    assert state["next"]["phase"] == "calibration"
    assert state["done"] == 0
    submit_phase(db, "calibration")
    assert progress(db, "practice-reviewer")["next"]["screen"] == "Quality Review"
    assert calibration_feedback(db)["human"]["reviewed"] == 0
    db.reveal_phase("calibration")
    assert calibration_feedback(db)["human"]["reviewed"] == 4
    assert findings(db)["human"]["reviewed"] == 0
    assert progress(db, "practice-reviewer")["next"]["phase"] == "main-01"


def test_waiting_repeats_suggest_other_work_without_exposing_answers(db):
    submit_phase(db, "calibration")
    db.reveal_phase("calibration")
    submit_phase(db, "main-01")
    db.schedule_repeats(count=2)
    state = progress(db, "practice-reviewer")
    assert state["next"]["phase"] == "main-02"
    assert state["done"] == 12
    assert state["total"] == 28
    assert not {"reference", "score", "trial_id", "path"} & set(state)


def test_research_bundle_does_not_unblind_pending_recordings(db):
    with zipfile.ZipFile(BytesIO(research_bundle(db))) as archive:
        assert {"annotations.csv", "findings.html", "metrics.json", "cases.csv"} <= set(archive.namelist())
        assert len(archive.read("annotations.csv").decode("utf-8-sig").splitlines()) == 1
        assert "DEMO_" not in archive.read("study_manifest.json").decode()


def test_redownload_preserves_blind_package_without_duplicate_assignments(db):
    original = export_package(db, "second-person", "calibration", 2)
    pid = db.one("SELECT id FROM packages")["id"]
    before = db.one("SELECT COUNT(*) AS n FROM assignments")["n"]
    rebuilt = package_bytes(db, pid)
    with zipfile.ZipFile(BytesIO(original)) as a, zipfile.ZipFile(BytesIO(rebuilt)) as b:
        assert a.read("manifest.json") == b.read("manifest.json")
        for name in a.namelist():
            if name.startswith("audio/"):
                assert a.read(name) == b.read(name)
        assert not any(word in b.read("manifest.json").decode() for word in ('"reference"', '"score"', 'DEMO_'))
    assert db.one("SELECT COUNT(*) AS n FROM assignments")["n"] == before
