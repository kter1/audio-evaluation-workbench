from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import APP
import soundfile as sf

from lab.forensics import (RATE, analyze, decision, diagnostics, mono16, read_audio,
                           report_html, save_report, windows)


def blob(x, rate=RATE):
    stream = BytesIO()
    sf.write(stream, x, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def tone(seconds=3, rate=RATE):
    return (.2*np.sin(2*np.pi*1000*np.arange(round(seconds*rate))/rate)).astype(np.float32)


def test_diagnostics_known_sine_and_gain():
    _, m, warnings = diagnostics(tone()[:, None], RATE)
    assert m["rms_dbfs"] == pytest.approx(20*np.log10(.2/np.sqrt(2)), abs=.01)
    assert m["spectral_centroid_hz"] == pytest.approx(1000, abs=1)
    assert m["clipped_sample_fraction"] == 0
    assert m["active_frame_fraction"] == 1
    assert not warnings
    _, quieter, _ = diagnostics(tone()[:, None]*.5, RATE)
    assert quieter["rms_dbfs"]-m["rms_dbfs"] == pytest.approx(-6.0206, abs=.001)
    assert quieter["spectral_centroid_hz"] == pytest.approx(m["spectral_centroid_hz"])


@pytest.mark.parametrize("seconds", [0.5, 2.5, 6, 6.2, 8.49, 8.5, 12, 13, 120])
def test_windows_cover_exactly_once_without_short_tails(seconds):
    x = tone(seconds)
    spans = windows(x)
    assert spans[0][0] == 0 and spans[-1][1] == len(x)
    assert sum(b-a for a, b in spans) == len(x)
    assert all(spans[i][1] == spans[i+1][0] for i in range(len(spans)-1))
    if len(x) >= 2.5*RATE:
        assert all(b-a >= 2.5*RATE for a, b in spans)


def test_weighted_aggregation_and_disagreement():
    metrics = diagnostics(tone()[:, None], RATE)[1]
    scores = [{"start": 0, "end": 6, "synthetic_score": .9}, {"start": 6, "end": 9, "synthetic_score": .1}]
    d = decision(scores, metrics, 9)
    assert d["synthetic_score"] == pytest.approx((6*.9+3*.1)/9)
    assert d["binary_model_decision"] == "synthetic"
    assert d["assessment"] == "Inconclusive"
    assert "Windows disagree" in d["abstention_reasons"][0]
    with pytest.raises(ValueError):
        decision([{**scores[0], "synthetic_score": float("nan")}], metrics, 6)


def test_silence_clipping_and_stereo_cancellation_abstain():
    scores = [{"start": 0, "end": 3, "synthetic_score": .99}]
    for x in [np.zeros((3*RATE, 1)), np.ones((3*RATE, 1)), np.stack([tone(), -tone()], axis=1)]:
        _, m, warnings = diagnostics(x, RATE)
        assert warnings
        assert decision(scores, m, 3)["assessment"] == "Inconclusive"


def test_input_decode_limits_and_resampling():
    audio, rate, source = read_audio(blob(tone(rate=44100), 44100))
    x = mono16(audio, rate)
    assert len(x) == 3*RATE
    assert source["duration_seconds"] == 3
    assert np.sqrt(np.mean(x**2)) == pytest.approx(.2/np.sqrt(2), abs=.001)
    for bad in [b"corrupt", blob(tone(.1)), blob(tone(121))]:
        with pytest.raises(ValueError):
            read_audio(bad)


class TestDetector:
    __test__ = False
    manifest = {"repository": "test-only", "revision": "fixture", "files": {}}
    parameters = 0

    def score(self, x):
        # Deterministic fake only for testing pipeline mechanics, never study evidence.
        p = float(np.clip(.8+float(x.mean()), 0, 1))
        return {"synthetic_score": p, "human_score": 1-p, "logits": [0., 1.]}


def test_pipeline_reproducibility_and_export(tmp_path):
    payload = blob(tone(9))
    first, x = analyze(payload, TestDetector(), robustness=True)
    second, _ = analyze(payload, TestDetector(), robustness=True)
    assert first["windows"] == second["windows"]
    assert first["robustness"] == second["robustness"]
    assert first["source"]["sha256"] == second["source"]["sha256"]
    assert len(first["robustness"]) == 3
    for row in first["robustness"]:
        baseline = next(s for s in first["windows"] if s["start"] == row["start"])
        assert row["baseline_synthetic_score"] == baseline["synthetic_score"]
    path = save_report(first, tmp_path)
    assert json.loads(path.read_text()) == first
    first["warnings"].append("<script>alert(1)</script>")
    html = report_html(first)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert len(x) == 9*RATE


def test_forensics_ui_reference_does_not_touch_study(db, monkeypatch):
    from streamlit.testing.v1 import AppTest
    safe = db.one("SELECT id FROM recordings WHERE subset='progress' LIMIT 1")["id"]
    with db.conn:
        db.set_setting("forensic_examples", [safe])
        db.conn.execute("UPDATE recordings SET duration=3 WHERE id=?", (safe,))
    sf.write(db.one("SELECT path FROM recordings WHERE id=?", (safe,))["path"], tone(), RATE)
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Detector Lab").run()
    assert not app.exception
    assert app.title[0].value == "Detector Lab"
    app.radio[0].set_value("Reference demonstration").run()
    assert not app.exception
    assert any(b.label == "Analyze recording" for b in app.button)
    assert not db.rows("SELECT * FROM annotation_revisions")


def test_full_report_ui_and_downloads(db, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import lab.forensics_ui as ui
    safe = db.one("SELECT id FROM recordings WHERE subset='progress' LIMIT 1")["id"]
    with db.conn:
        db.set_setting("forensic_examples", [safe])
        db.conn.execute("UPDATE recordings SET duration=3 WHERE id=?", (safe,))
    sf.write(db.one("SELECT path FROM recordings WHERE id=?", (safe,))["path"], tone(), RATE)
    monkeypatch.setenv("AUDIO_LAB_DB", str(db.path))
    monkeypatch.setattr(ui, 'detector', lambda kind: TestDetector())
    monkeypatch.setattr(ui, 'save_report', lambda r: save_report(r, tmp_path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value('Detector Lab').run()
    app.radio[0].set_value('Reference demonstration').run()
    next(b for b in app.button if b.label == 'Analyze recording').click().run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ['Detector evidence', 'Signal laboratory', 'Sensitivity tests', 'Method & provenance']
    assert any(m.label == 'Synthetic-class score' for m in app.metric)
    assert len(list(tmp_path.glob('*.json'))) == 1
    assert not db.rows('SELECT * FROM annotation_revisions')


def test_model_manifest_cannot_override_pinned_digest(tmp_path):
    from lab.forensics import MODEL_ID, REVISION, verify_model
    import hashlib
    files = {}
    for name in ['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md']:
        content = b'untrusted replacement'
        (tmp_path/name).write_bytes(content)
        files[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    (tmp_path/'manifest.json').write_text(json.dumps({'repository': MODEL_ID, 'revision': REVISION, 'files': files}))
    with pytest.raises(ValueError, match='checksum verification'):
        verify_model(tmp_path)


def test_explain_reports_each_check_against_its_limit():
    from lab.forensics import explain
    report, _ = analyze(blob(tone(9)), TestDetector(), robustness=True)
    why = explain(report)
    status = {c["check"]: c["status"] for c in why["checks"]}
    assert status["No stereo cancellation"] == "n/a"
    assert all(v == "pass" for k, v in status.items() if k != "No stereo cancellation")
    assert why["verdict"] == "Flagged as synthetic: score 0.80 is at or above the 0.50 threshold and all 7 applicable reliability checks passed."
    short = explain(analyze(blob(tone(2)), TestDetector())[0])
    assert {c["check"]: c["status"] for c in short["checks"]}["Long enough"] == "fail"
    assert {c["check"]: c["status"] for c in short["checks"]}["Stable under perturbations"] == "n/a"
    assert short["verdict"].startswith("Inconclusive: the model leans synthetic") and "shorter than 2.5 seconds" in short["verdict"]


def test_detector_threshold_moves_vote_band_and_explanation():
    from lab.forensics import explain

    class Strict(TestDetector):
        threshold = .85
    report, _ = analyze(blob(tone(9)), Strict())
    assert report["policy"]["binary_threshold"] == .85
    assert (report["policy"]["human_below"], report["policy"]["synthetic_above"]) == (.7, 1.0)
    assert report["assessment"]["binary_model_decision"] == "human"
    assert report["assessment"]["assessment"] == "Inconclusive"
    assert "0.85" in explain(report)["verdict"]
    assert 'y1="51.00"' in report_html(report)  # dotted threshold line at 0.85
