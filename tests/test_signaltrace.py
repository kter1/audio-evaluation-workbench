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

client = TestClient(server.app, base_url='http://127.0.0.1')


def test_local_security_headers_and_dns_rebinding_guard():
    response = client.get('/')
    assert response.headers['x-frame-options'] == 'DENY'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['cache-control'] == 'no-store'
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert client.get('/api/status', headers={'host': 'attacker.example'}).status_code == 400


def test_chunked_upload_limit_applies_without_content_length(monkeypatch):
    monkeypatch.setattr(server, 'MAX_BYTES', 64)
    boundary = b'--test-boundary\r\nContent-Disposition: form-data; name="file"; filename="voice.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    def body():
        yield boundary
        yield b'x' * 20000
        yield b'\r\n--test-boundary--\r\n'
    response = client.post('/api/inspect', content=body(), headers={'content-type': 'multipart/form-data; boundary=test-boundary'})
    assert response.status_code == 413


def test_foreign_origin_cannot_upload():
    response = client.post('/api/inspect', files={'file': ('voice.wav', b'x')}, headers={'origin': 'https://attacker.example'})
    assert response.status_code == 400


def test_annotation_example_has_exact_audio_coverage_and_honest_review_status():
    page = client.get('/samples/annotation-example.html')
    assert page.status_code == 200
    assert 'Listening review has not been performed' in page.text
    assert 'Transcript with playback' in page.text
    assert 'Perceived origin: uncertain' not in page.text
    assert '/samples/annotation-example.html' in client.get('/').text
    raw = client.get('/samples/annotation-demo.wav').content
    audio, rate = sf.read(BytesIO(raw))
    assert rate == 16000 and audio.shape == (480000,)
    result = client.get('/samples/annotation-demo.json').json()
    assert sha256(raw).hexdigest() == result['sample_sha256']
    assert result['listening_review_completed'] is False
    assert result['detector_inference_run'] is False
    assert result['perceptual_annotation']['perceived_origin'] == 'uncertain'
    assert result['perceptual_annotation']['confidence'] is None
    assert result['reference']['license'] == 'CC BY 4.0'
    assert result['transcription']['sample_sha256'] == result['sample_sha256']
    assert len(result['transcription']['words']) == 79
    assert len(result['transcript_review']) == 6
    assert result['transcript_review'][0]['model_text'] == 'heddy surl'
    assert result['transcript_review'][0]['reference_text'] == 'hetty sorrel'
    intervals = [r for r in result['annotations'] if r['kind'] == 'interval_review']
    assert [(r['start'], r['end']) for r in intervals] == [(s, s+5) for s in range(0, 30, 5)]
    for row in result['annotations']:
        assert 0 <= row['start'] < row['end'] <= 30
        assert row['review_status'] in {'measured', 'context_review'}
    assert 'start,end,kind,observation' in client.get('/samples/annotation-demo.csv').text
    assert client.get('/samples/annotation-example.js').status_code == 200
    assert 'model_text,reference_text' in client.get('/samples/annotation-review.csv').text


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


def test_inspect_preserves_native_channel_measurements_when_mono_cancels():
    rate = 48000
    x = (1.2*np.sin(2*np.pi*12000*np.arange(rate)/rate)).astype(np.float32)
    stream = BytesIO()
    sf.write(stream, np.stack([x, -x], axis=1), rate, format="WAV", subtype="FLOAT")
    result = client.post("/api/inspect", files={"file": ("opposite-phase.wav", stream.getvalue())})
    assert result.status_code == 200, result.text
    data = result.json()
    original = data["metrics"]["source_measurements"]
    assert original["sample_rate_hz"] == rate
    assert len(original["channels"]) == 2
    assert original["arithmetic_mixdown"]["complete_cancellation"]
    assert all(c["sample_peak_amplitude"] == pytest.approx(1.2) for c in original["channels"])
    assert all(c["over_full_scale_fraction"] == pytest.approx(.5) for c in original["channels"])
    assert data["metrics"]["rms_dbfs"] == -240
    assert any("mono mixing masks" in warning for warning in data["warnings"])
    assert data["metrics"]["analysis_representation"]["sample_rate_hz"] == 16000


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


@pytest.mark.parametrize('kind', ['short', 'silent', 'canceled', 'clipped'])
def test_quality_preflight_skips_model_and_preserves_inconclusive(monkeypatch, kind):
    t = np.arange(3*16000)/16000
    x = .2*np.sin(2*np.pi*440*t)
    if kind == 'short': x = x[:8000]
    elif kind == 'silent': x = np.zeros_like(x)
    elif kind == 'canceled': x = np.column_stack((x, -x))
    else: x = np.ones_like(x)
    def forbidden():
        raise AssertionError('Model must not load for rejected recording')
    monkeypatch.setattr(server, 'fusion_model', forbidden)
    monkeypatch.setattr(server, 'detectors_available', lambda: True)
    response = client.post('/api/infer', files={'file': ('voice.wav', audio_blob(x))})
    assert response.status_code == 200
    result = response.json()
    assert result['model_executed'] is False
    assert result['assessment']['assessment'] == 'Inconclusive'
    assert result['assessment']['synthetic_score'] is None
    assert result['assessment']['binary_model_decision'] is None
    assert result['assessment']['abstention_reasons']
    assert result['windows'] == [] and result['links'] is None


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
