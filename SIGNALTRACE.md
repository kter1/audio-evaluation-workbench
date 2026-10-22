# SignalTrace — Audio Evidence Studio

SignalTrace is a local waveform-first audio investigation instrument.
It highlights **measured** signal changes, synchronizes each timestamped
observation with playback, and can optionally add window-level synthetic
speech model evidence using the repository's existing two-model fusion.

## Launch on macOS

Open Terminal in the repository, then run:

```sh
git fetch origin
git switch feature/signaltrace-studio
git pull --ff-only origin feature/signaltrace-studio
sh run.sh
```

Open **http://127.0.0.1:8501**. Stop with Ctrl+C.
The first launch installs a few lightweight dependencies if needed.

At startup, SignalTrace opens a randomly selected audio clip from the local recording catalog. If a research database has no playable media, a separate local SQLite sample library is automatically created under `data/` with four reproducible **non-speech** WAV samples. Use **Random sample** to choose another clip; the sample source is always labeled.

Click **Try signal demo** for a generated, **non-speech** audio fixture containing
known quiet, frequency-change and saturated intervals. The demo proves only
that the acoustic visualization and detection rules work, not that any voice
is synthetic. Upload your own audio to investigate a recording. Accepted source
formats depend on the local decoder, typically WAV, FLAC, MP3, OGG and AIFF.
Limits: 256 MiB, 0.5–600 seconds (10 minutes), up to 8 channels at 8–192 kHz, and 60 million decoded samples per source. Very high-rate multichannel files may reach the sample cap earlier.

### Optional synthetic-speech inference

Signal analysis, region playback, JSON evidence export, and the **Benchmark
lab** work without neural model weights. To install the original frozen fusion:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_pellav2.py
.venv/bin/python scripts/setup_antideepfake.py
```

The pretrained checkpoint downloads can be large. The combined detector
inherits **non-commercial** model licensing restrictions. Neither model is
downloaded automatically at startup. Model scores are uncalibrated class
outputs, not verified probabilities of authenticity or accurate edit
boundaries. A detected acoustic change is likewise *not* proof of tampering.

Historical research comes from docs/detector-evaluation.json: 320 source clips
(200 human, 120 synthetic) reused in four audio conditions, not 1,280
independent recordings. Benchmarks and live user recordings remain separate.

### Legacy application

Use `sh run-legacy.sh` for the original annotation and evaluation Streamlit
application. All original detectors, evaluation receipts, and study code are
preserved.

## Implementation

- `studio/dist/` — self-contained HTML/CSS/Canvas/Web Audio front end;
  no external JS CDN or Node build required.
- `signaltrace_server.py` — localhost-only Starlette API and audio analyzer.
- `lab/signal_events.py` — versioned deterministic acoustic and model markers.
- `lab/sample_catalog.py` — random local SQLite-backed sample selection with a transparent generated fallback.
- `lab/forensics.py`, `lab/detectors.py`, `lab/fusion.py` — existing research engine.
- `tests/test_signaltrace.py` — API, marker and benchmark regression tests.

The studio intentionally focuses on two destinations: **Analyze** and
**Benchmark lab**. Evidence exports include timestamped markers and local-session
analyst notes; audio files are not exported or saved by the signal-only path.
After optional model inference, separate research reports may be saved under
data/analyses, consistent with the original forensics engine.

## Acoustic evidence instruments

The main SignalTrace view uses distinct SVG glyphs and marker treatments for
clipping/saturation, low-energy spans, sharp level changes, spectral-band shifts,
short signal gaps, impulse-like discontinuities, baseline/DC offsets, limited
high-frequency energy, stereo channel imbalance and phase inversion. Optional
model instruments distinguish fusion threshold crossings, component score
separation and sensitivity to controlled signal transformations.

Every marker is an **observation with a rule, numeric measurement, timestamps,
limitations and a next check**. These rules do not prove manipulation, identify
a speaker, or supply causal explanations for the neural network. In particular,
a naturally low-pitched tone may have limited high-band energy, a conversational
pause may resemble a gap, and regular speech can produce abrupt transitions.
Findings can overlap because the detectors describe independent signal properties.
The waveform always links to the original media; no speculative labels are
silently added.

The original acoustic rule set is in `lab/signal_events.py`; additional local
quality detectors live in `lab/signal_intelligence.py`. Regression fixtures
exercise short gaps, stereo polarity and channel imbalance. This is engineering
screening, not independently validated anomaly classification.
