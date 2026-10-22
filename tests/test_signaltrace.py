import asyncio
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

import signaltrace_server as server
from lab.signal_events import acoustic_markers, model_markers

client = TestClient(server.app)


def audio_blob(audio, rate=16000):
    stream = BytesIO()
    sf.write(stream, audio, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def test_homepage_and_assets():
    home = client.get("/")
    assert home.status_code == 200
    assert "SignalTrace" in home.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
    assert "data-page" in home.text


def test_signal_demo_is_explicitly_non_speech_and_reproducible():
    resp = client.get("/api/demo")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    raw = resp.content
    a = client.post("/api/inspect", files={"file": ("demo.wav", raw, "audio/wav")})
    assert a.status_code == 200, a.text
    result = a.json()
    assert result["duration"] == pytest.approx(12, abs=.01)
    assert result["sha256"] == sha256(raw).hexdigest()
    assert result["peaks"]
    assert 0 < len(result["markers"]) <= 40
    assert {"clipping", "low_energy", "spectral_change"} & {
        m["kind"] for m in result["markers"]
    }
    b = client.post("/api/inspect", files={"file": ("demo.wav", raw, "audio/wav")})
    assert [m["id"] for m in result["markers"]] == [m["id"] for m in b.json()["markers"]]
    for m in result["markers"]:
        assert 0 <= m["start"] < m["end"] <= result["duration"]
        assert m["method"] and m["threshold"] and m["limitations"] and m["next_check"]


def test_corrupt_audio_and_missing_file_fail_closed():
    assert client.post("/api/inspect", files={"file": ("broken.wav", b"garbage", "audio/wav")}).status_code == 400
    assert client.post("/api/inspect", data={}).status_code == 400
    assert client.post("/api/inspect", files={"file": ("huge.wav", b"a"*(server.MAX_BYTES+1))}).status_code == 400


def test_origin_restriction():
    demo = client.get("/api/demo").content
    request = client.post("/api/inspect", files={"file": ("t.wav", demo)},
                          headers={"Origin": "https://attacker.example"})
    assert request.status_code == 400


def test_known_acoustic_intervals_are_not_verdicts():
    rate = 16000
    x = np.zeros(5*rate, dtype=np.float32)
    x[rate:2*rate] = 1.0
    flags = acoustic_markers(x, rate, "0"*64)
    assert any(f["kind"] == "clipping" for f in flags)
    assert any(f["kind"] == "low_energy" for f in flags)
    assert all("synthetic" not in f["title"].lower() for f in flags)
    with pytest.raises(ValueError):
        acoustic_markers(np.array([np.nan]*rate), rate, "x"*64)


def test_model_window_boundaries_are_preserved():
    report = {
        "source": {"sha256":"a"*64},
        "policy":{"binary_threshold":0.6},
        "windows":[{"start":4.0,"end":8.0,"synthetic_score":0.91,
                    "components":{"pellav2":0.1,"antideepfake":0.92}}],
        "robustness":[],
    }
    flags = model_markers(report)
    assert len(flags) == 2
    assert all(flag["start"] == 4 and flag["end"] == 8 for flag in flags)
    assert {f["kind"] for f in flags} == {"synthetic_score","component_separation"}


def test_benchmark_receipt_is_not_fresh_inference():
    r = client.get("/api/benchmark")
    assert r.status_code == 200
    data = r.json()
    assert data["base_clips"] == 320
    assert data["human_clips"] == 200
    assert data["synthetic_clips"] == 120
    assert len(data["conditions"]) == 4
    for condition in data["conditions"].values():
        assert condition["fusion"]["synthetic_total"] == 120
        assert condition["fusion"]["human_total"] == 200
    assert len(data["rows"]) == 1280


def test_missing_model_returns_explanation(monkeypatch):
    monkeypatch.setattr(server, "detectors_available", lambda: False)
    r = client.post("/api/infer", files={"file": ("clip.wav", client.get("/api/demo").content)})
    assert r.status_code == 409
    assert "checkpoints" in r.json()["error"].lower()


def test_report_rejects_invalid_token():
    assert client.get("/api/reports/../../json").status_code in (404, 400)
    assert client.get("/api/reports/" + "0"*64 + "/json").status_code == 404


def test_sample_catalog_random_endpoint_and_playable_media():
    catalog = client.get("/api/samples")
    assert catalog.status_code == 200
    items = catalog.json()["samples"]
    assert len(items) >= 4
    assert all("extension" in item for item in items)
    chosen = client.get("/api/samples/random")
    assert chosen.status_code == 200
    entry = chosen.json()
    assert entry["id"] in {item["id"] for item in items}
    recording = client.get(f"/api/samples/{entry['id']}/audio")
    assert recording.status_code == 200
    assert len(recording.content) > 100
    result = client.post("/api/inspect",
        files={"file": ("startup" + entry["extension"], recording.content)})
    assert result.status_code == 200, result.text
    assert result.json()["duration"] >= 0.5


def test_long_recordings_support_ten_minutes_and_reject_over_limit():
    from lab.forensics import read_audio, MAX_SECONDS, MAX_BYTES
    assert MAX_SECONDS == 600
    assert MAX_BYTES >= 100 * 1024 * 1024
    # Do not perform 10 minutes of neural inference in the regression test.
    one = audio_blob(np.zeros(600 * 16000, dtype=np.float32))
    decoded, rate, metadata = read_audio(one)
    assert rate == 16000
    assert metadata["duration_seconds"] == 600
    assert decoded.shape[0] == 600 * 16000
    too_long = audio_blob(np.zeros(601 * 16000, dtype=np.float32))
    with pytest.raises(ValueError, match="600"):
        read_audio(too_long)


def test_generated_fallback_has_no_arbitrary_human_synthetic_label():
    from lab.sample_catalog import generated_entries, load, resolve
    choices = generated_entries()
    assert len(choices) >= 4
    for entry in choices:
        assert entry["speech"] is False
        assert entry["source"] == "generated_signal"
        assert len(load(resolve(entry["id"]))) > 100


def test_extended_quality_detectors_on_controlled_audio():
    from lab.signal_intelligence import extra_acoustic_markers, stereo_markers
    rate = 16000
    t = np.arange(rate * 7)/rate
    audio = (.15*np.sin(2*np.pi*250*t)).astype(np.float32)
    audio[rate*2:rate*2+3200] = 0
    results = extra_acoustic_markers(audio, "f"*64)
    assert "signal_gap" in {x["kind"] for x in results}
    for entry in results:
        assert 0 <= entry["start"] < entry["end"] <= 7
        assert entry["method"] and entry["threshold"] and entry["limitations"]


def test_stereo_quality_controls_are_distinct():
    from lab.signal_intelligence import stereo_markers
    rate = 16000
    t = np.arange(rate*3)/rate
    left = (.20*np.sin(2*np.pi*430*t)).astype(np.float32)
    inverted = np.stack([left, -left], axis=1)
    assert "phase_inversion" in {x["kind"] for x in stereo_markers(inverted,rate,"b"*64)}
    imbalanced = np.stack([left,left*.04],axis=1)
    assert "channel_imbalance" in {x["kind"] for x in stereo_markers(imbalanced,rate,"c"*64)}
    assert stereo_markers(left[:,None],rate,"d"*64) == []


def test_distinct_svg_glyphs_are_registered():
    script = client.get("/app.js").text
    names = ["clipping","level_change","spectral_change","low_energy",
             "signal_gap","impulse","dc_offset","bandwidth_limit",
             "phase_inversion","channel_imbalance",
             "synthetic_score","component_separation","sensitivity"]
    for name in names:
        assert name + ":" in script
    assert "evidenceIcon(m.kind)" in script
