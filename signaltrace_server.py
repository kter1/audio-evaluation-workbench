"""Local-only SignalTrace API; no hosted audio inference or invented model scores."""
from __future__ import annotations

import asyncio
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "studio" / "dist"
RECEIPT = ROOT / "docs" / "detector-evaluation.json"
ANALYSES = ROOT / "data" / "analyses"
MAX_BYTES = 256 * 1024 * 1024
MODEL_LOCK = asyncio.Lock()


def detectors_available():
    from lab.detectors import installed
    return installed("fusion")


def benchmark_data():
    record = json.loads(RECEIPT.read_text())
    models = {"fusion": record["test"], **{
        key: {"by_channel": value["by_channel"]}
        for key, value in record["comparison"].items()
    }}
    conditions = {}
    for key, values in record["test"]["by_channel"].items():
        conditions[key] = {
            name: {
                "human_total": model["by_channel"][key]["reals"]["n"],
                "human_flagged": model["by_channel"][key]["reals"]["flagged_synthetic"],
                "synthetic_total": model["by_channel"][key]["fakes"]["n"],
                "synthetic_flagged": model["by_channel"][key]["fakes"]["flagged_synthetic"],
            }
            for name, model in models.items()
        }
    return {
        "version": record["version"], "base_clips": record["test"]["summary"]["n"],
        "human_clips": record["test"]["summary"]["human"],
        "synthetic_clips": record["test"]["summary"]["synthetic"],
        "threshold": record["threshold"], "conditions": conditions,
        "rows": [{
            "id": x["id"], "channel": x["channel"], "label": x["label"],
            "source": x.get("source"), "generator": x.get("generator"),
            "score": x["score"], "assessment": x.get("assessment"),
            "pellav2": x.get("components",{}).get("pellav2"),
            "antideepfake": x.get("components",{}).get("antideepfake"),
        } for x in record["test"]["rows"]],
        "limitations": [
            "Archived evaluation data—not a fresh classification of an uploaded file.",
            "The same 320 clips were tested in four transformed channel conditions; variants are correlated.",
            "English read speech and three open-source synthetic generators only.",
            "AntiDeepfake model weights have non-commercial research license conditions.",
            "Model-score thresholds and abstention rules are not calibrated probabilities of authentic speech.",
        ],
    }


def waveform_peaks(samples, max_bins=4000):
    x = np.asarray(samples, dtype=np.float32)
    stride = max(1, int(np.ceil(len(x) / max_bins)))
    peaks = []
    for i in range(0, len(x), stride):
        part = x[i:i+stride]
        peaks.append([round(float(part.min()),4), round(float(part.max()),4)])
    return peaks


def inspect_bytes(raw):
    from lab.forensics import read_audio, diagnostics, RATE
    from lab.signal_events import acoustic_markers
    from lab.signal_intelligence import extra_acoustic_markers, stereo_markers
    decoded, rate, metadata = read_audio(raw)
    samples, measurements, warnings = diagnostics(decoded, rate)
    return {
        "sha256": metadata["sha256"],
        "duration": metadata["duration_seconds"],
        "sample_rate": rate, "channels": metadata["channels"],
        "format": metadata["format"], "bytes": metadata["bytes"],
        "metrics": measurements, "warnings": warnings,
        "peaks": waveform_peaks(samples),
        "markers": sorted([*acoustic_markers(samples, RATE, metadata["sha256"]),
                           *extra_acoustic_markers(samples, metadata["sha256"]),
                           *stereo_markers(decoded, rate, metadata["sha256"])],
                          key=lambda item: (item["start"], item["kind"])),
        "method": "Deterministic local signal analysis; no speech identity or authenticity claims.",
    }


@lru_cache(maxsize=1)
def fusion_model():
    from lab.detectors import load
    return load("fusion")


def infer_bytes(raw):
    from lab.forensics import analyze, explain, save_report, read_audio, diagnostics, quality_blockers
    from lab.signal_events import model_markers
    audio, rate, source = read_audio(raw)
    _, measurements, warnings = diagnostics(audio, rate)
    blockers = quality_blockers(measurements, source["duration_seconds"])
    if blockers:
        return {
            "id": None, "sha256": source["sha256"], "model_executed": False,
            "assessment": {"assessment": "Inconclusive", "synthetic_score": None,
                           "binary_model_decision": None, "abstention_reasons": blockers,
                           "interpretation": "Recording quality prevents a reliable assessment; model inference was skipped."},
            "threshold": None, "model_name": None, "checks": [], "windows": [],
            "markers": [], "warnings": warnings, "links": None,
        }
    report, _ = analyze(raw, fusion_model(), robustness=True)
    target = save_report(report)
    return {
        "id": target.stem, "sha256": report["source"]["sha256"], "model_executed": True,
        "assessment": report["assessment"], "threshold": report["policy"]["binary_threshold"],
        "model_name": report["model"].get("name"), "checks": explain(report)["checks"],
        "windows": [{
            "start": x["start"], "end": x["end"],
            "score": x["synthetic_score"], "components": x.get("components", {})
        } for x in report["windows"]],
        "markers": model_markers(report),
        "warnings": report["warnings"],
        "links": {"json": f"/api/reports/{target.stem}/json",
                  "html": f"/api/reports/{target.stem}/html"},
    }


def sample_audio():
    """Public-domain generated signal fixture (not speech or benchmark data)."""
    rate = 16000
    x = np.zeros(12*rate, dtype=np.float32)
    t = np.arange(rate*12) / rate
    # Three obvious sections: changing tone, quiet interval and saturated tone.
    a, b = int(.45*rate), int(4.0*rate)
    x[a:b] = (.25*np.sin(2*np.pi*220*t[a:b])).astype(np.float32)
    a, b = int(4.0*rate), int(6.5*rate)
    x[a:b] = (.55*np.sin(2*np.pi*830*t[a:b])).astype(np.float32)
    a, b = int(7.5*rate), int(11.2*rate)
    x[a:b] = np.clip(2.8*np.sin(2*np.pi*440*t[a:b]),-.999, .999).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, x, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


async def upload_bytes(request):
    origin = request.headers.get("origin")
    if origin and not re.fullmatch(r"https?://(?:localhost|127[.]0[.]0[.]1)(?::[0-9]+)?", origin):
        raise ValueError("Cross-origin uploads to the local service are disabled.")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError as exc:
            raise ValueError("Invalid request size.") from exc
        if size > MAX_BYTES + 16384:
            raise ValueError("Upload exceeds the 256 MB audio limit.")
    form = await request.form(max_files=1, max_fields=1)
    file = form.get("file")
    if not hasattr(file, "read"):
        raise ValueError("Select an audio recording.")
    raw = await file.read(MAX_BYTES+1)
    await file.close()
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Select an audio file no larger than 256 MB.")
    return raw


async def status(request):
    return JSONResponse({
        "ready": detectors_available(),
        "model": "PellaV2 + NII AntiDeepfake · fusion v4",
        "setup": [
            ".venv/bin/python -m pip install -r requirements.txt",
            ".venv/bin/python scripts/setup_pellav2.py",
            ".venv/bin/python scripts/setup_antideepfake.py",
        ],
        "license": "Research-only use; combined detector inherits non-commercial restrictions.",
    })


async def benchmark(request):
    try:
        return JSONResponse(benchmark_data())
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


async def samples(request):
    from lab.sample_catalog import catalog
    return JSONResponse({"samples": catalog()})


async def random_sample(request):
    from lab.sample_catalog import choose
    entry = choose()
    return JSONResponse({k: v for k, v in entry.items() if k != "_path"})


async def sample_file(request):
    from lab.sample_catalog import load, resolve
    entry = resolve(request.path_params["identifier"])
    if entry is None:
        return JSONResponse({"error": "Sample unavailable."}, status_code=404)
    try:
        raw = await run_in_threadpool(load, entry)
        return Response(raw, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except (OSError, ValueError):
        return JSONResponse({"error": "Sample file no longer available."}, status_code=404)


async def demo(request):
    return Response(sample_audio(), media_type="audio/wav",
                    headers={"Cache-Control":"no-store"})


async def inspect(request):
    try:
        raw = await upload_bytes(request)
        return JSONResponse(await run_in_threadpool(inspect_bytes, raw))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def infer(request):
    if not detectors_available():
        return JSONResponse({"error": "Verified model checkpoints are missing. Install both models first."}, status_code=409)
    try:
        raw = await upload_bytes(request)
        async with MODEL_LOCK:
            result = await run_in_threadpool(infer_bytes, raw)
        return JSONResponse(result)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def download(request):
    token = request.path_params["token"]
    kind = request.path_params["kind"]
    if not re.fullmatch(r"[a-f0-9]{64}", token) or kind not in {"json", "html"}:
        return JSONResponse({"error": "Report not found."}, status_code=404)
    path = ANALYSES / (token + ".json")
    if not path.is_file():
        return JSONResponse({"error": "Report not found."}, status_code=404)
    if kind == "json":
        return FileResponse(path, media_type="application/json", filename="signaltrace-model-report.json")
    from lab.forensics import report_html
    try:
        return Response(report_html(json.loads(path.read_text())), media_type="text/html",
                        headers={"Content-Disposition": 'attachment; filename="signaltrace-model-report.html"'})
    except (ValueError, KeyError, OSError):
        return JSONResponse({"error": "Cannot read report."}, status_code=500)


async def homepage(request):
    return FileResponse(DIST / "index.html")


app = Starlette(routes=[
    Route("/api/status", status),
    Route("/api/benchmark", benchmark),
    Route("/api/demo", demo),
    Route("/api/samples", samples),
    Route("/api/samples/random", random_sample),
    Route("/api/samples/{identifier:str}/audio", sample_file),
    Route("/api/inspect", inspect, methods=["POST"]),
    Route("/api/infer", infer, methods=["POST"]),
    Route("/api/reports/{token:str}/{kind:str}", download),
    Mount("/", app=StaticFiles(directory=DIST, html=True)),
])
