# SignalTrace — AI Voice Check

Upload a voice recording and get one clear result: **AI voice indicated**,
**No clear AI signal**, or **Inconclusive**, with a short explanation.
Processing runs locally. The current detector combines PellaV2 and AntiDeepfake;
its accuracy has not been established as world-class.

## Run

From the project folder:

```sh
sh run.sh
```

Open [SignalTrace](http://127.0.0.1:8501). Stop with Ctrl+C. After updating Python
code, restart the server; refresh the page after frontend changes. The first
launch installs lightweight dependencies if needed.

1. Upload an English voice recording.
2. Detection starts automatically when the local models are installed.
3. Read the result and its reasons. Listen or open technical evidence if useful.

Use at least 2.5 seconds of speech. Supported inputs depend on the decoder,
typically WAV, FLAC, MP3, OGG and AIFF. Limits are 256 MiB, 0.5–600 seconds,
1–8 channels at 8–192 kHz, and 60 million decoded source samples.

Recordings that fail existing duration, activity, clipping or stereo-cancellation
checks return Inconclusive before loading the detector. A model result that
changes materially under the existing perturbation check is also Inconclusive.
These checks do not establish that an input is speech or verify its language.

## Install the voice detector

If the screen says setup is required, run:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_pellav2.py
.venv/bin/python scripts/setup_antideepfake.py
```

Refresh the page afterward. Checkpoint downloads can be large. The combined
model inherits non-commercial licensing restrictions. Opening the app does not
automatically download models. Model scores are not calibrated authenticity
probabilities; music, unfamiliar generators, replay and recording changes remain
limitations. A negative result does not certify human origin: short AI inserts
in a longer recording can be missed.

## Details when needed

Playback and recording evidence are collapsed by default. Technical details
retain the raw model score, threshold, window results and report downloads.
Recording details retain native channel measurements, signal warnings, markers,
analyst notes and evidence export. Signal changes do not establish AI generation
or edit boundaries. Generated non-speech demos remain under those signal tools;
no sample loads automatically.

**About the detector → Validation history** shows the archived benchmark. It is
not a fresh evaluation: 320 source clips were reused across four conditions.
The legacy Streamlit app remains available through `sh run-legacy.sh`.

Research records remain available without occupying the main product workflow:
[baseline audit](docs/baseline-methods-audit.md),
[evaluation contract](docs/evaluation-contract-v1.json),
[source measurements](docs/source-measurements.md),
[method research](docs/method-research.md),
[VAD development](docs/vad-development-study.md),
[local runtime](docs/vad-runtime-study.md),
[gain/noise stress](docs/vad-stress-study.md), and
[channel/short-context stress](docs/vad-channel-stress-study.md).

The current [core-detector progress report](docs/core-detector-progress.md)
records direct testing on five newer TTS systems and a reliability correction:
a human indication requires corroboration from both component detectors.
Conflicting evidence is inconclusive; raw fused scores remain available in details.


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
