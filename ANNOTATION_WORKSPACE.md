# ◉ Audio Review Lab

A blind speech-annotation workspace for telling real voices from AI voice clones. I use it to measure my own listening: how accurate and how consistent my labels are, which cues I rely on, and where I get fooled. Only after a batch is locked do I compare my judgments with the dataset's truth and with a modern deepfake detector.

## Blind by design

Annotations are only useful if a model or an answer key hasn't already influenced them. The workflow enforces that:

- **Judge first.** Reference labels, detector scores, file names, and earlier answers stay hidden during review.
- **Freeze, then reveal.** Submitting freezes the original judgment. A batch is revealed only when every clip in it is submitted.
- **Corrections never overwrite.** Later corrections and adjudications are stored separately, next to the original.
- **Consistency is measured, not assumed.** Twenty clips come back as blind repeats at least 48 hours later, and agreement with my earlier self is scored (Cohen's kappa).
- **Independent review is possible.** A blind package (neutral WAVs, no metadata) can go to a second reviewer, whose returned labels are compared before adjudication.
- **No detector during review.** The Detector Lab scores only audio I upload, and it warns while any clip in the workspace is still blind.

## My results

> ⏳ **In progress.** Numbers appear here only from revealed, exported work. Nothing is estimated or prefilled.

| Work | Clips | Result |
|---|---|---|
| Listening challenge: real voices vs. modern clones | 20 (10 real, 10 clones from 3 generators) | ⏳ fakes that fooled me · detector caught |
| Calibration batch (learning, excluded from results) | 20 | ⏳ |
| Main batch 1 | 80 | ⏳ accuracy among decided · uncertainty rate |
| Blind repeats, ≥ 48 h later | 20 | ⏳ self-agreement (kappa) |
| Which cues held up | all revealed clips | ⏳ |
| Investigated cases | 5 | ⏳ |

## What I record for every clip

Defined in the [annotation guide](docs/annotation-guide.md) (versioned; v1.1 registered before any review began):

- **Intelligibility:** clear, partly intelligible, unintelligible, or no speech.
- **Channel and environment:** background noise, compression artifacts, reverb or echo, music or hold audio, clipping, dropouts, distortion, overlap.
- **Perceptual cues:** what I heard that leans synthetic (unnatural prosody, flat emotion, metallic timbre, missing breaths, odd stress, splices, too-clean sound) or human (natural breaths, hesitations, consistent room acoustics).
- **Perceived origin:** human, synthetic, or **uncertain**, which is a valid answer with a required explanation.
- **Confidence (1–5),** **timestamped observations** (for example, "1.18–1.36 s: abrupt silence mid-word"), and an **escalation** with the open question when a case falls outside the guide.

After reveal, **Quality Review** filters for detector mistakes, disagreements, escalations, and uncertain calls, and turns a clip into a short case study: observation, possible explanation, recommended next check. **Findings & Export** produces spreadsheet-ready CSVs (annotations, revision history, cases, adjudications), metrics, and a printable report. [SQL examples](docs/queries.sql) query the same SQLite records.

## The detector, used only after reveal

The Detector Lab fuses two independently trained speech-deepfake models: [pellav2](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector) and NII's [AntiDeepfake](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake). Every result explains itself: each model's vote, the threshold, and pass/fail reliability checks for duration, level, clipping, and stability under noise.

On a frozen test of 320 new clips from 73 speakers, scored once:

| Channel | Clones caught | Real voices wrongly flagged |
|---|---|---|
| Clean | 120/120, including 60/60 from a generator held out of development | 0/200 |
| MP3 64 kbps | 120/120 | 3/200 (1.5%) |
| Telephone, 8 kHz | 120/120 | 5/200 (2.5%) |
| Added background noise | 119/120 | 10/200 (5.0%) |

**What this means for listening work:** compressed and noisy genuine speech is where models struggle most. The detector's false alarms concentrate there, and so do the hardest calls for a human ear. That is why the guide treats channel artifacts as interference, not as evidence of synthesis. Full method, failures, and limits are in [the evaluation](docs/model-evaluation.md) and [technical method](docs/forensics-method.md). Scope: English read speech, three open generators, and simulated channels. This is not a test of commercial voices or live calls.

## Data handling

- Everything runs locally on `127.0.0.1`. No audio is sent to an online service.
- Audio, model weights, and work databases live under a git-ignored `data/` folder and are never committed.
- Real speech comes from public research corpora ([ASVspoof 2021](https://www.asvspoof.org/index2021.html), ODC-By; [LibriSpeech](https://www.openslr.org/12), CC BY 4.0). Voice clones were generated locally from public audiobook readers, for research only, and are not distributed.
- Some components are licensed for non-commercial research only: AntiDeepfake weights (CC BY-NC-SA 4.0), and the XTTS-v2 generator (Coqui Public Model License) used to make test clips.

## Security

- **Local only.** The app binds to `127.0.0.1`, has upload limits (50 MB, 120 s, 24 million samples), and makes no network calls during review or inference.
- **Model files are verified before use.** Every checkpoint and config has a pinned SHA-256 checked before loading. PyTorch weights load with `weights_only=True`, AntiDeepfake with safetensors, and Hugging Face loaders never run remote code. Both detector ports are parity-checked against their official code.
- **Safe parsing.** Archives are read by exact expected member names (no path traversal); SQL is parameterized; HTML in reports and the UI is escaped; CSV exports neutralize spreadsheet formulas.
- **Dependencies are pinned and audited.** `requirements.lock` reported no known vulnerabilities in `pip-audit` on 2026-10-19, after upgrading torch, transformers, Streamlit, and Pillow. A regression check confirmed the published detector results did not change.

## Study status

The study separates personal listening judgments from automated detector results. Completed annotation accuracy and independent reviewer agreement have not been established. Detector claims must be read alongside the frozen manifests, receipts, and documented limitations in [`docs/`](docs/).

## Run it

Requires Python 3.11–3.13 (tested on 3.13, Apple Silicon). From this folder:

```sh
./run-legacy.sh
```

Open [localhost:8501](http://localhost:8501). The first launch creates a virtual environment, installs `requirements.lock`, and opens a clearly labeled practice workspace (generated tones, arbitrary labels). Switch between the research study, listening challenge, and practice workspaces in the sidebar. Drafts survive restarts, and **Save & skip** moves on without losing work.

The Detector Lab needs two one-time downloads (about 1.26 GB each, checksum-verified). Inference then runs offline on CPU:

```sh
.venv/bin/python scripts/setup_pellav2.py
.venv/bin/python scripts/setup_antideepfake.py
```

## Research setup

1. Read [the annotation guide](docs/annotation-guide.md) and [study protocol](docs/study-protocol.md).
2. Fetch the official ASVspoof 2021 metadata, scores, and source mappings (~36 MB; no audio):

```sh
.venv/bin/python -m lab fetch-metadata
```

3. Download the [official ASVspoof 2021 LA audio archive](https://zenodo.org/records/4837263) (~7.8 GB), verify its published MD5 (`2abee34d8b0b91159555fc4f016e4562`), keep its notices, and extract it.
4. Register the dataset and create the source-separated sample:

```sh
.venv/bin/python -m lab --db data/study.sqlite import-asvspoof --audio-dir /absolute/path/to/ASVspoof2021_LA_eval/flac
.venv/bin/python -m lab --db data/study.sqlite prepare-study --reviewer your-name
```

The sample has **20 calibration + 80 + 100 main + 100 fresh validation = 300 unique recordings**. Each batch is class-balanced, and no source speaker or text repeats across batches. Unreadable or corrupt audio is excluded and recorded, never counted as reviewed.

The listening challenge is built from the frozen modern-voice test after the detector evaluation:

```sh
.venv/bin/python -m lab --db data/challenge-v4.sqlite prepare-challenge --reviewer your-name
```

## Study operations

Complete calibration and reveal it, revise the guide if needed (as a new version, before starting the next batch), then complete main batch 1. Schedule 20 blind repeats, wait at least 48 hours, complete them, and only then reveal. Assign any independent review **before** reveal. Quality Review handles all of this in the app; the CLI equivalents are:

```sh
.venv/bin/python -m lab --db data/study.sqlite schedule-repeats --count 20
.venv/bin/python -m lab --db data/study.sqlite reveal --phase main-01
.venv/bin/python -m lab --db data/study.sqlite export --directory exports/study
```

See [the protocol](docs/study-protocol.md) for independent reviewer packages and the [walkthrough](docs/walkthrough.md) for a three-minute demo.

## Verification

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Boundaries

A trusted, local, single-operator app with no authentication. Reviewer isolation uses separate databases and sanitized media packages, not logins. The owner can inspect their own SQLite data, so blind mode is a study discipline, not protection against a dishonest owner. The project evaluates audio only. It does not establish fraud, identify speakers, or evaluate any vendor's production system. Confidence is self-reported, disagreement is not automatically annotation error, and independent agreement remains **not measured** until another person actually returns reviews.

## Sources and attribution

- [ASVspoof 2021](https://www.asvspoof.org/index2021.html) and its [Logical Access dataset](https://zenodo.org/records/4837263), [baseline scores and evaluation package](https://github.com/asvspoof-challenge/2021/tree/main/eval-package), and [source mappings](https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz); ODC-By attribution, official notices retained.
- [LibriSpeech](https://www.openslr.org/12), CC BY 4.0.
- Detectors: [pellav2](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector) (MIT), [NII AntiDeepfake](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake) (CC BY-NC-SA 4.0), [AASIST](https://github.com/clovaai/aasist) (MIT); vendored notices are in `lab/vendor/`.
- Generators used for test clips: [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT), [F5-TTS](https://github.com/SWivid/F5-TTS), [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (Coqui Public Model License).

ASVspoof source groups use the official VCTK speaker ID for genuine and converted speech and normalized input text for TTS. Identical spoken text across human and TTS recordings is not fully observable from this mapping.
