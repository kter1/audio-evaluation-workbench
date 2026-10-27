# Documentation

Reports are written in Markdown. The `*.json` files beside them are
machine-readable receipts (manifests, hashes, per-clip scores, and acquisition
records) that the reports cite. Receipts are kept byte-for-byte as generated so
their hashes stay verifiable; please don't hand-edit them.

## Using the tools

| Document | What it covers |
| --- | --- |
| [App guide](app-guide.md) | Running SignalTrace, installing the detector, reading results and evidence markers |
| [Annotation workspace](annotation-workspace.md) | The blind review app: workflow, study design, research setup, and CLI |
| [Annotation guide v1.1](annotation-guide.md) | Labeling definitions: intelligibility, interference tags, cues, origin, confidence, escalation |
| [Study protocol](study-protocol.md) | Batch order, blind repeats, independent reviewer packages, and adjudication |
| [Walkthrough](walkthrough.md) | Three-minute demonstration script for the annotation workspace |
| [Example report](forensics-example.html) | Sample exported detector report |
| [SQL examples](queries.sql) | Queries against the annotation database |

## Detector evaluation

| Document | What it covers |
| --- | --- |
| [Detector evaluation](model-evaluation.md) | Protocols v2–v4, including the frozen 320-clip test and earlier results that missed the bar |
| [Detector method](forensics-method.md) | Model ports, parity checks, fusion, reliability checks, and perturbation tests |
| [Core detector progress](core-detector-progress.md) | Screening against five newer TTS systems and the policy 1.1 human-corroboration rule |
| [Capability investigation](detector-capability-update.md) | Complementary WavLM detector, broader regression, and short-insert results |
| [Complementary calibration](complementary-calibration.md) | Retrospective calibration probe for the complementary detector |

## Rejected approaches

| Document | What it covers |
| --- | --- |
| [Alternative classifier](core-classifier-comparison.md) | Third classifier that failed the modern-voice screen |
| [Noise preprocessing](core-noise-improvement.md) | Spectral reduction, low-pass, and MetricGAN experiments |
| [RawNetLite](core-rawnetlite-decision.md) | Compact noise-augmented classifier |

## Data and methods

| Document | What it covers |
| --- | --- |
| [Baseline methods audit](baseline-methods-audit.md) | Baseline integrity audit and the prospective evaluation contract |
| [Source measurements](source-measurements.md) | Native-rate, per-channel recording measurements |
| [Method research](method-research.md) | Corpus qualification gate and development method comparisons |
| [Corpus roadmap](corpus-acquisition-roadmap.md) | Local corpus inventory and acquisition plan |
| [ASVspoof 5 pilot](asvspoof5-pilot.md) | 40-recording unscored pilot and its integrity audit |
| [ASVspoof 5 expansion](asvspoof5-expanded.md) | 900 unscored recordings from 600 declared speakers |

## Speech activity (VAD) studies

| Document | What it covers |
| --- | --- |
| [Development study](vad-development-study.md) | Annotated Silero vs. WebRTC comparison, review packets, agreement, and adjudication |
| [Runtime study](vad-runtime-study.md) | Local CPU latency and resource use |
| [Stress study](vad-stress-study.md) | Quiet-input and constructed-noise robustness |
| [Channel stress study](vad-channel-stress-study.md) | Codec, telephone channel, and short-context robustness |

## Project records

| Document | What it covers |
| --- | --- |
| [Research log](research-log.md) | Chronological summary of research milestones and decisions |
| [Security review](security-review.md) | Local service hardening, dependency audit, and remaining limits |
| [Implementation verification](validation.md) | Software and data-intake checks for the annotation study |
