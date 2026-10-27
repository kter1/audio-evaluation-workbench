# Implementation verification · October 19, 2026

This receipt describes software and data-intake checks. It is **not** a completed personal annotation study or independent quality review.

## Software checks

- **33 tests passed** in an isolated virtual environment created from the pinned lockfile (Python 3.13.7, Apple Silicon).
- `pip check`: no broken requirements.
- All three Streamlit screens rendered; draft save/resume, frozen submission, and fractional-second observation submission were exercised through Streamlit's application test framework.
- Tests cover corrupt media, blind projections and exports, 48-hour repeat access, immutable guidelines, independent handoff/conflicting imports, sample reproducibility, precise audio slicing, spreadsheet formula injection, missing-score accounting, and hand-calculated metrics.
- A strict FLAC fallback regression test verifies bit-exact PCM and rejects a mismatched embedded PCM checksum.
- Documented SQL queries, shell syntax, and Python compilation checks passed.

## Official data intake

- Imported **181,566** official LA metadata rows and **181,566** published LFCC-GMM scores.
- Frozen operating threshold uses only **16,464 progress trials**: 1,676 human and 14,788 spoof recordings. Threshold: **1.155591**, with higher scores supporting human speech. This is a study operating point, not a production recommendation.
- Verified the **7,763,666,128-byte** official archive against published MD5 `2abee34d8b0b91159555fc4f016e4562`.
- Archive SHA-256: `d8b37ed81026aa0885e4f699421e9f80d5d14261531d0a0e976e79dd031e3c79`.
- Inspected the actual source mapping format: source VCTK ID for genuine/converted recordings, normalized input text for TTS. Cross-type text identity is not fully observable.

## Prepared local study

The untracked local research database contains **300 assigned recordings**, **300 distinct source groups**, and **zero source-group overlap with progress calibration**. All 300 assigned files decoded successfully and have original-file SHA-256 checksums. SQLite integrity check: `ok`.

| Phase | Human references | Spoof references | Total |
|---|---:|---:|---:|
| Calibration | 10 | 10 | 20 |
| First main batch | 40 | 40 | 80 |
| Second main batch | 50 | 50 | 100 |
| Fresh validation | 50 | 50 | 100 |

Preparation seed: `20261019`. Study manifest SHA-256: `52a7481217291d4f3ca515c84ae9b54acfcdebdc2e4d91ce5b9afe1b0a5d5e45`.

**Path correction:** real-data verification exposed original FLAC that SoundFile could inspect but not fully decode. A pinned FFmpeg fallback was added without rewriting source audio. Fallback output is accepted only when sample count, sample rate, channel count, and embedded original PCM checksum match. **41 candidate recordings** failed strict decoding checks and were excluded during sampling. This is a compatibility/integrity filter, not a human/synthetic or fraud judgment; findings must acknowledge this selection limitation.

The selected 300 study files plus four separate reference examples remain extracted. The verified source archive is cached locally for reproducibility; audio, databases, and exports are excluded from Git. A fresh clone must follow the [research setup](annotation-workspace.md#research-setup) instructions.

At implementation handoff: **zero personal annotations, zero independent reviewer judgments, and zero authored research case studies**. The software does not fabricate these. The local demo remains separate and explicitly labeled as generated non-speech fixtures.


## Forensic inference extension — October 19, 2026

- Added a separate default Forensics Lab with genuine local inference, time-window logits/scores, STFT/Welch diagnostics, three controlled perturbations, and JSON/HTML exports.
- Installed and checksum-verified the 1,262,859,296-byte pinned safetensors checkpoint; counted 315,701,634 parameters. No remote model code or hosted inference API.
- Full automated suite: **57 passed**. Dependency consistency: no broken requirements.
- Four actual reference recordings scored, then scored again with `scripts/check_detector.py`. Repeated inference on the first file matched within 1e-6 relative tolerance. Exact outputs and limitations are in `detector-validation.json`.
- Raw threshold decisions: two synthetic detected, one human correctly classified, one human falsely flagged. Three policy abstentions; the sole decided case was wrong. The four examples are not a representative or independent benchmark. Large perturbation score changes are disclosed.
- Live UI completed real inference and rendered detector evidence, signal charts, and technical downloads. Resolved a Plotly argument compatibility warning against pinned Streamlit 1.50.
- Research database retained **300 assignments and zero annotation events**. Integrity check: `ok`.


## Default detector correction and fresh evaluation

- Replaced the unsuitable 2.05-second default demo with four examples selected by provenance and duration before scoring: 2.90, 4.46, 4.04, and 4.166 seconds.
- Compared the old Wav2Vec2 detector and pinned official AASIST on 40 fixed selection cases. Corrected pipeline results: 27/40 versus 36/40.
- Corrected tiny-tail handling in the AASIST adapter. Retired the initial test, froze the updated model choice, and used 40 fresh source groups for confirmation: **38/40 correct**, one human false positive and one synthetic miss. Reliability policy: **15 abstentions, 24/25 correct decided cases**. No test threshold fitting.
- All four preselected demos now match their references and pass the reliability policy; three sensitivity probes per demo remain stable (largest change <0.0025).
- **71 automated tests passed**, including window-boundary coverage, full-context tails, source exclusion, class mapping mechanics, count metrics/Wilson interval, and UI rendering. Dependency consistency passed.
- Full methodology, the integration correction, and evaluation limits are recorded in `model-evaluation.md`. Historical receipts remain available; these small local results are not a real-world accuracy guarantee.
