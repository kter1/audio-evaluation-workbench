# SignalTrace app guide

Upload a voice recording and get one clear result: **AI voice indicated**,
**No clear AI signal**, or **Inconclusive**, with a short explanation.
Processing runs locally. The detector combines PellaV2 and NII AntiDeepfake; see
the [README](../README.md#results) for what it has and has not been tested on.

## Run

From the project folder:

```sh
sh run.sh
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) and stop the server with
Ctrl+C. The first launch creates `.venv/` and installs the lightweight app
dependencies.

1. Upload an English voice recording.
2. Detection starts automatically once the local models are installed.
3. Read the result and its reasons. Listen, or open the technical evidence, if useful.

Use at least 2.5 seconds of speech. Supported formats depend on the decoder,
typically WAV, FLAC, MP3, OGG, and AIFF. Limits are 256 MiB, 0.5–600 seconds,
1–8 channels at 8–192 kHz, and 60 million decoded source samples.

Recordings that fail the duration, activity, clipping, or stereo-cancellation
checks return Inconclusive before the detector loads. A model result that
changes materially under the perturbation check is also Inconclusive. These
checks do not establish that an input is speech or verify its language.

## Install the voice detector

If the status badge says **Detector setup required**, run:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_pellav2.py
.venv/bin/python scripts/setup_antideepfake.py
```

Each checkpoint is about 1.3 GB and is verified against a pinned SHA-256 before
it loads. Refresh the page afterward. Opening the app never downloads models.
The combined model inherits non-commercial licensing restrictions.

Model scores are not calibrated authenticity probabilities. Music, unfamiliar
generators, replayed audio, and recording changes remain limitations. A negative
result does not certify human origin: short AI inserts in a longer recording can
be missed.

## Details when needed

Playback and recording evidence are collapsed by default. **Technical details**
show the raw model score, threshold, per-window results, and report downloads.
**Recording details** show native channel measurements, signal warnings, markers,
analyst notes, and evidence export. Signal changes do not establish AI generation
or edit boundaries. Generated non-speech demos live under the signal tools; no
sample loads automatically.

**About the detector → Validation history** shows the archived benchmark. It is
not a fresh evaluation: 320 source clips were reused across four conditions.

Since policy 1.1, a human indication requires both component detectors to agree.
Conflicting evidence is Inconclusive, and the raw fused score stays available in
the technical details. See [core detector progress](core-detector-progress.md).

## Acoustic evidence markers

The waveform view uses distinct glyphs and marker styles for clipping and
saturation, low-energy spans, sharp level changes, spectral-band shifts, short
signal gaps, impulse-like discontinuities, baseline (DC) offsets, limited
high-frequency energy, stereo channel imbalance, and phase inversion. Optional
model markers show fusion threshold crossings, component score separation, and
sensitivity to controlled signal transformations.

Every marker is an **observation with a rule, a numeric measurement, timestamps,
limitations, and a next check**. These rules do not prove manipulation, identify
a speaker, or explain the neural network's decision. A naturally low-pitched
voice may have limited high-band energy, a conversational pause may resemble a
gap, and ordinary speech can produce abrupt transitions. Findings can overlap
because the rules measure independent signal properties. The waveform always
links to the original media, and no speculative labels are added.

The acoustic rules are in [`lab/signal_events.py`](../lab/signal_events.py), with
additional quality checks in [`lab/signal_intelligence.py`](../lab/signal_intelligence.py).
Regression fixtures exercise short gaps, stereo polarity, and channel imbalance.
This is engineering screening, not independently validated anomaly classification.

## Development notes

Restart the server after changing Python code; refresh the page after changing
files in `studio/dist/`. The front end has no build step.
