import pytest

from conftest import APP
from streamlit.testing.v1 import AppTest


def test_three_screens_render_without_errors(db, monkeypatch):
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Your review workspace"  # human review first
    app.sidebar.radio[0].set_value("Detector Lab").run()
    assert app.title[0].value == "Detector Lab"
    app.sidebar.radio[0].set_value("Workspace").run()
    assert app.title[0].value == "Your review workspace"
    for name in ("Review", "Quality Review", "Findings & Export"):
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception
        assert app.title[0].value == name


def test_draft_save_and_resume_in_ui(db, monkeypatch):
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Review").run()
    assert not app.exception
    app.text_area[0].set_value("No speech in this fixture; uncertain.")
    next(b for b in app.button if b.label == "Save draft").click().run()
    assert not app.exception
    assert db.one("SELECT COUNT(*) AS n FROM annotation_revisions WHERE event='draft'")["n"] == 1
    # AppTest retains stale form nodes after an explicit st.rerun; refresh its tree.
    app.run()
    next(b for b in app.button if b.label == "Submit & next").click().run()
    assert not app.exception
    assert db.one("SELECT COUNT(*) AS n FROM assignments WHERE submitted_at IS NOT NULL")["n"] == 1


def test_timestamped_observation_survives_editor_submission(db, monkeypatch):
    import json
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Review").run()
    aid = app.selectbox[1].value
    app.session_state[f"notes_{aid}False"] = {"edited_rows": {}, "added_rows": [{"start": 0.123, "end": 0.456, "note": "Observed interval in a test tone"}], "deleted_rows": []}
    app.text_area[0].set_value("No speech in fixture; origin uncertain.")
    next(b for b in app.button if b.label == "Submit & next").click().run()
    assert not app.exception
    original = db.one("SELECT payload FROM annotation_revisions WHERE assignment_id=? AND event='submitted'", (aid,))
    assert original is not None
    assert json.loads(original["payload"])["observations"] == [{"start": 0.123, "end": 0.456, "note": "Observed interval in a test tone"}]


def test_save_and_skip_keeps_draft_and_changes_clip(db, monkeypatch):
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Review").run()
    original = app.selectbox[1].value
    app.text_area[0].set_value("Need another listen later.")
    next(b for b in app.button if b.label == "Save & skip").click().run()
    app.run()
    assert not app.exception
    assert app.selectbox[1].value != original
    assert db.blind_review(original)["annotation"]["rationale"] == "Need another listen later."
    assert not db.assignment(original)["submitted_at"]


def test_guided_warmup_gives_feedback_without_recording_study_reviews(db, monkeypatch):
    safe = db.one("SELECT id FROM recordings WHERE subset='progress' LIMIT 1")["id"]
    with db.conn:
        db.set_setting("learning_examples", [safe])
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Workspace").run()
    next(t for t in app.toggle if t.label == "Warm up with reference examples").set_value(True).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Reveal this reference").click().run()
    assert not app.exception
    assert any("Dataset reference:" in notice.value for notice in app.info)
    assert not db.rows("SELECT * FROM annotation_revisions")
