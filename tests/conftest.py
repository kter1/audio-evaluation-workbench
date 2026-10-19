from pathlib import Path

import pytest

from lab.demo import create_demo
from lab.store import Store

APP = str(Path(__file__).resolve().parents[1] / "app.py")  # AppTest resolves relative paths from the test file


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_demo(path)
    store = Store(path)
    yield store
    store.close()


def payload(**changes):
    result = {"intelligibility": "no speech", "tags": [], "cues": [], "judgment": "uncertain", "confidence": 2,
              "observations": [{"start": 0.12, "end": 0.34, "note": "Generated test tone, no audible speech"}],
              "rationale": "Non-speech fixture; origin cannot be judged", "escalated": False, "escalation_reason": ""}
    result.update(changes)
    return result


def submit_phase(db, phase):
    for a in db.rows("SELECT id FROM assignments WHERE role='primary' AND phase=? ORDER BY ordinal", (phase,)):
        db.save(a["id"], payload(), "submitted")
