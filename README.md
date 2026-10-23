# SignalTrace · AI Voice Check

A local audio deepfake detection and research project exploring synthetic speech, recording quality, and reliable evaluation. Built as a personal portfolio project to demonstrate interest in audio annotation and research quality. This is an independent project, unaffiliated with Pindrop.

![SignalTrace main screen: upload a recording for a local AI voice check](docs/images/signaltrace-main.png)

## Try it locally

```sh
git clone https://github.com/kter1/audio-evaluation-workbench.git
cd audio-evaluation-workbench
sh run.sh
```

Open **http://127.0.0.1:8501**. The default launcher serves the redesigned SignalTrace app. Python 3.13 is used for verification. To install the optional local detector checkpoints, follow [detector setup](SIGNALTRACE.md#install-the-voice-detector). The app does not download models automatically.

Upload a recording to get **AI voice indicated**, **No clear AI signal**, or **Inconclusive**. Recording playback, quality measurements, model evidence, analyst notes, and report exports are available when needed. Audio processing runs locally; audio is not uploaded to a hosted detection API.

## What the project demonstrates

- **Careful interpretation:** uncertain or unsuitable recordings can produce an inconclusive result. A negative result does not certify human origin.
- **Targeted investigation:** tests examine noise, compression, telephone bandwidth, model disagreement, and short synthetic inserts.
- **Dataset integrity:** source hashes, provenance, duplicate checks, and declared-speaker separation are recorded in reproducible research reports.
- **Annotation tooling:** the earlier blind review workspace supports structured observations, timestamped notes, escalation, and spreadsheet-ready exports. See [annotation workspace](ANNOTATION_WORKSPACE.md) and [annotation guide](docs/annotation-guide.md). Personal review results and independent reviewer agreement remain unmeasured.

## Evidence and limitations

The current app combines PellaV2 and NII AntiDeepfake. Its archived 320-source regression test detected 120/120 synthetic recordings in clean audio and 119/120 with added noise; genuine false alerts rose from 0/200 clean to 10/200 noisy. These reused research samples do not establish performance on real customer calls or unseen commercial generators.

A complementary WavLM model remains experimental. The latest external set contains **900 unscored recordings from 600 declared speakers**, with reserved calibration and evaluation partitions. Integrity checks passed; unknown model training exposure prevents calling it an independently unseen benchmark.

- [Detector capability findings and known failures](docs/detector-capability-update.md)
- [External corpus acquisition and qualification](docs/asvspoof5-expanded.md)
- [Research progress and rejected approaches](docs/core-detector-progress.md)
- [App setup and interpretation](SIGNALTRACE.md)
- [Archived detector evaluation](docs/model-evaluation.md)

Short AI inserts can be missed. Model scores are not calibrated authenticity probabilities. The project does not establish fraud, authenticate identity, evaluate video, or reproduce any vendor's proprietary technology. It is a research prototype, with no claim of world-class accuracy.

## Verification

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
node --check studio/dist/app.js
node --test tests/frontend-evidence.test.cjs
```

Publication checks: **372 Python tests and 6 frontend tests passed**. Full model inference additionally requires separately acquired checkpoints and data.

## Privacy, licensing, and authorship

Audio corpora, model weights, work databases, exports, and personal notes stay outside Git. Dataset and model acquisition scripts preserve source provenance and license notices. Several model/data components have non-commercial or research restrictions; consult [setup](SIGNALTRACE.md), individual acquisition reports, and [third-party notices](lab/vendor/) before use or redistribution.

Automated tests and research reports are distinguished from personal listening work; no completed human annotations or independent agreement results are claimed. Earlier study documentation and attribution are preserved in [the annotation workspace guide](ANNOTATION_WORKSPACE.md).
