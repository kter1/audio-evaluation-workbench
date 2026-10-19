from datetime import datetime, timedelta, timezone
import json

import pytest

import lab.store as state
from lab.store import Store
from lab.reporting import annotation_csv, findings, history_csv, manifest_export, report_html
from conftest import payload, submit_phase


def first(db, phase="calibration"):
    return db.one("SELECT * FROM assignments WHERE role='primary' AND phase=? ORDER BY ordinal LIMIT 1", (phase,))


def test_save_restart_freeze_and_separate_correction(db):
    a = first(db)
    db.save(a["id"], payload(), "draft")
    other = Store(db.path)
    assert other.blind_review(a["id"])["annotation"]["rationale"] == payload()["rationale"]
    other.close()
    db.save(a["id"], payload(), "submitted")
    assert db.blind_review(a["id"])["annotation"] is None
    with pytest.raises(ValueError, match="frozen"):
        db.save(a["id"], payload(), "draft")
    with pytest.raises(ValueError, match="revealed"):
        db.correction(a["id"], payload(judgment="human"))
    for row in db.rows("SELECT id FROM assignments WHERE phase='calibration' AND submitted_at IS NULL"):
        db.save(row["id"], payload(), "submitted")
    db.reveal_phase("calibration")
    db.correction(a["id"], payload(judgment="human"))
    history = db.rows("SELECT event,payload FROM annotation_revisions WHERE assignment_id=? ORDER BY id", (a["id"],))
    assert [r["event"] for r in history] == ["draft", "submitted", "correction"]
    original = db.one("SELECT payload FROM annotation_revisions WHERE assignment_id=? AND event='submitted'", (a["id"],))
    assert json.loads(original["payload"])["judgment"] == "uncertain"


@pytest.mark.parametrize("changes,match", [
    ({"rationale": ""}, "uncertain"),
    ({"escalated": True, "escalation_reason": ""}, "reason"),
    ({"observations": [{"start": -1, "end": 1, "note": "bad"}]}, "timestamps"),
    ({"observations": [{"start": 1, "end": 0.5, "note": "bad"}]}, "timestamps"),
    ({"observations": [{"start": 0, "end": float('nan'), "note": "bad"}]}, "timestamps"),
    ({"confidence": True}, "Confidence"),
    ({"judgment": "fraud"}, "classification"),
])
def test_validation_rejects_invalid_complete_reviews(db, changes, match):
    with pytest.raises(ValueError, match=match):
        db.save(first(db)["id"], payload(**changes), "submitted")
    assert not first(db)["submitted_at"]


def test_draft_allows_unfinished_uncertainty_and_escalation(db):
    a = first(db)
    db.save(a["id"], payload(rationale="", escalated=True), "draft")
    assert db.blind_review(a["id"])["annotation"]["escalated"]


def test_corrupt_and_changed_media_cannot_be_submitted(db):
    a = first(db)
    path = db.one("SELECT path FROM recordings WHERE id=?", (a["recording_id"],))["path"]
    from pathlib import Path
    original = Path(path).read_bytes()
    Path(path).write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        db.save(a["id"], payload(), "submitted")
    assert not db.assignment(a["id"])["submitted_at"]
    assert db.queue("practice-reviewer")[0]["media_error"]
    Path(path).write_bytes(original)
    db.save(a["id"], payload(), "submitted")
    assert db.assignment(a["id"])["submitted_at"]


def test_blind_projection_and_exports_have_no_truth_or_prior_answers(db):
    submit_phase(db, "calibration")
    for a in db.queue("practice-reviewer"):
        blind = db.blind_review(a["id"])
        assert not {"trial_id", "reference", "score", "path", "source_group", "origin_id", "codec", "attack"} & set(blind)
        if a["submitted_at"]:
            assert blind["annotation"] is None
    assert not db.revealed_rows()
    assert len(annotation_csv(db).decode("utf-8-sig").splitlines()) == 1
    assert len(history_csv(db).decode("utf-8-sig").splitlines()) == 1
    assert "DEMO_" not in report_html(db)
    assert "DEMO_" not in json.dumps(manifest_export(db))


def test_repeats_wait_48_hours_and_block_reveal(db, monkeypatch):
    submit_phase(db, "main-01")
    with pytest.raises(ValueError, match="repeats"):
        db.reveal_phase("main-01")
    assert db.schedule_repeats(count=2) == 2
    repeats = db.rows("SELECT * FROM assignments WHERE role='repeat'")
    with pytest.raises(ValueError, match="submitted"):
        db.reveal_phase("main-01")
    for a in repeats:
        with pytest.raises(ValueError, match="48 hours"):
            db.review_audio(a["id"])
        blind = db.blind_review(a["id"])
        assert blind["annotation"] is None and "origin_id" not in blind
    future = (datetime.now(timezone.utc) + timedelta(hours=49)).isoformat()
    monkeypatch.setattr(state, "now", lambda: future)
    for a in repeats:
        db.save(a["id"], payload(), "submitted")
    db.reveal_phase("main-01")
    result = findings(db)
    assert result["submitted_unique"] == 8
    assert result["within_reviewer"]["pairs"] == 2
    assert result["within_reviewer"]["agreement"] == 1
    assert result["independent"]["agreement"] is None
    assert not result["milestone_100"]  # Never count a demo as hiring evidence.
    with pytest.raises(ValueError, match="already scheduled"):
        db.schedule_repeats(count=2)


def test_calibration_excluded_and_adjudication_does_not_change_metrics(db):
    submit_phase(db, "calibration")
    db.reveal_phase("calibration")
    submit_phase(db, "main-02")
    db.reveal_phase("main-02")
    before = findings(db)
    assert before["human"]["reviewed"] == 8
    assert before["submitted_unique"] == 12
    rid = first(db, "main-02")["recording_id"]
    db.adjudicate(rid, "owner", "human", "Fixture adjudication test only")
    db.add_case(rid, "Test case", "A tone was heard", "Cause not established", "Get speech evidence")
    assert findings(db) == before
    assert "Test case" in report_html(db)


def test_missing_scores_counted_separately(db):
    submit_phase(db, "main-02")
    db.reveal_phase("main-02")
    rid = first(db, "main-02")["recording_id"]
    with db.conn:
        db.conn.execute("DELETE FROM detector_scores WHERE recording_id=?", (rid,))
    result = findings(db)
    assert result["missing_scores"] == 1
    assert result["detector"]["scored"] == 7
    assert result["human"]["reviewed"] == 8


def test_guidelines_immutable_and_started_phase_cannot_be_reassigned(db):
    db.add_guideline("2.0", "Revised definitions", "validation")
    assert db.one("SELECT guideline FROM assignments WHERE phase='validation'")["guideline"] == "2.0"
    with pytest.raises(ValueError, match="immutable"):
        db.add_guideline("2.0", "different")
    a = first(db, "main-02")
    db.save(a["id"], payload(), "draft")
    with pytest.raises(ValueError, match="started"):
        db.add_guideline("3.0", "Next revision", "main-02")
