# Detector evaluation

## Protocol v4: two-detector fusion on modern voices (current)

**Result: pre-set bar met on a fresh test.** The default detector fuses pellav2 with NII's AntiDeepfake MMS-300M. On 320 new clips from 73 speakers who appear in no earlier set, scored once, it caught every fake from the generator it never saw in development and wrongly flagged no clean real speech.

| Channel | Chatterbox | F5-TTS | **XTTS-v2 (held out)** | Real speech wrongly flagged | Bar |
|---|---|---|---|---|---|
| Clean | 30/30 | 30/30 | **60/60** | **0/200** (0%; 95% CI 0–1.9%) | ≤2% ✓ |
| MP3 64 kbps | 30/30 | 30/30 | **60/60** | **3/200** (1.5%; 0.5–4.3%) | ≤5% ✓ |
| Telephone 8 kHz µ-law | 30/30 | 30/30 | **60/60** | **5/200** (2.5%; 1.1–5.7%) | ≤5% ✓ |
| White noise, 20 dB SNR | 29/30 | 30/30 | **60/60** | **10/200** (5.0%; 2.7–9.0%) | ≤5% ✓ (at the limit) |

Fake bars: at least 95% per generator in every channel (at least 90% for held-out XTTS-v2).

**The same clips scored by each model alone** (real speech wrongly flagged, out of 200: clean / MP3 / telephone / noise):

| Model | Real false alarms | Fakes missed (of 120 per channel) |
|---|---|---|
| **Fusion (default)** | **0 / 3 / 5 / 10** | 0 / 0 / 0 / 1 |
| pellav2 alone, same tuning rule | 1 / 41 / 38 / 29 | 3 / 2 / 0 / 7 (incl. XTTS) |
| AntiDeepfake alone, same tuning rule | 1 / 1 / 2 / 15 | 0 / 0 / 0 / 3 (noisy Chatterbox) |
| pellav2 at its authors' 0.5 (v3 default) | 17 / 92 / 98 / 98 | 1 / 0 / 0 / 0 |

Each model covers the other's weakness: pellav2 catches fakes, AntiDeepfake keeps compressed real speech from being accused. Fused, false alarms on MP3 and telephone audio fall roughly 20–30× compared with v3.

### How it was built

1. **Second detector.** AntiDeepfake MMS-300M was ported from fairseq to `transformers` and checked against the official code in an isolated fairseq environment (largest logit difference 9.2e-5).
2. **Development data, already seen.** All 360 v3 clips (whose results were reported before v4 began) in clean, MP3, telephone, and 20 dB noise versions: 1,440 clip versions, scored per 4-second window by both models.
3. **Rule fixed before the development features existed** (`scripts/fit_fusion.py`): a 3-parameter logistic fusion; five-fold cross-validation grouped by speaker; threshold = highest value still flagging 98% of out-of-fold fakes; the bar above with ≤5% false alarms on every degraded channel.
4. **Development attempts, all recorded:**
   - Threshold-only on pellav2: missed (best option left ~11% MP3 false alarms). `data/evaluation/v4-calibration-failed.json`.
   - Fusion attempt 1 (3 parameters): passed every cell except noise, 11/180 real flagged (6.1%).
   - Fusion attempt 2 (noise-aware, 6 parameters, rule fixed first): 10/180 on noise (5.6%), also a miss.
   - As decided in advance by the project owner, attempt 1 was frozen anyway with its development miss recorded in `lab/fusion-v4.json`, and the fresh test was run once.
5. **Fresh test.** LibriSpeech dev-clean and dev-other (new speakers) and fakes cloned from dev-clean speakers, including XTTS-v2, held out from all development. Same matched design (each fake reads a sentence its speaker also recorded), the same Whisper transcription gate (30 excluded), frozen manifest SHA-256 `7e20d2ff56d415e17b75a8a0e9d942ea2ad4ed997184617b29b1fd0b5b87fe0c` (path-free copy: `v4-set-audit.json`).

Fused per window: z = 2.0425 + 0.3069·(pellav2 logit) + 0.9340·(AntiDeepfake fake−real logit); threshold 0.2096 on the mean window probability.

### Limitations

- **Noise is at the limit, not comfortably under it.** 10/200 on the test (CI up to 9%) and 6.1% in development.
- **Three open generators, English read speech.** ElevenLabs and other commercial voices, voice conversion, conversational and accented speech, and other languages were not tested.
- **Channels are simulated** (MP3 encode, G.711 µ-law, white noise), not recorded live calls.
- **Training overlap is unknown.** pellav2's training data is private and may include Chatterbox or F5-TTS. XTTS-v2 is the held-out check; one generator is a narrow check.
- **Non-commercial.** The fusion inherits AntiDeepfake's CC BY-NC-SA 4.0 license, and XTTS-v2 test clips were generated under Coqui's non-commercial CPML, accepted by the project owner for local research use; no generated audio is distributed.
- **Correlated clips.** Wilson intervals assume independent clips; clips share speakers and texts.

### Reproduce

```sh
.venv/bin/python scripts/setup_pellav2.py && .venv/bin/python scripts/setup_antideepfake.py
.venv/bin/python scripts/check_pellav2_parity.py
.venv/bin/python scripts/check_antideepfake_parity.py      # needs data/tts-envs/fairseq (Python 3.9, fairseq 862efab)
.venv/bin/python scripts/develop_fusion.py render && .venv/bin/python scripts/develop_fusion.py score
.venv/bin/python scripts/fit_fusion.py --model base --accept-miss
.venv/bin/python scripts/build_v4_set.py plan              # then generate, gate, channels, freeze
.venv/bin/python scripts/evaluate_v4.py test
```

Generator environments' exact packages are in `scripts/tts/*.lock`. The v3 receipt is `detector-evaluation-v3.json`.

**Dependency upgrade check (2026-10-19).** The v4 test was scored with torch 2.9.1 and transformers 4.57.3. For security fixes, the app was then upgraded to torch 2.14.1, transformers 5.20.2, and Streamlit 1.66.0. Both parity checks still pass, and `scripts/check_receipt_regression.py` re-scored 100 frozen test clip versions (25 per channel) on the new stack: largest score change 0.005 (0.24 on the logit scale, only far from the threshold) and **no verdict changes**. The receipt above is unchanged and still describes the shipped app.

---

## Protocol v3: modern voices, pellav2 alone

The question: does the lab catch modern, human-convincing cloned voices without accusing real people? AASIST, the previous default, was trained on 2019-era attacks. The new default is **pellav2** (XLS-R-300M, MIT), the strongest downloadable model on an independent modern-TTS benchmark.

**Pre-set bar** (fixed before any modern clip was scored): on the clean test split, flag at least 95% of fakes and at most 5% of real speech.

**Result: bar not met.** pellav2 caught every fake, but wrongly flagged 11.7% of real speech.

| Clean test split | pellav2 | AASIST (2019) | Wav2Vec2 |
|---|---|---|---|
| Fakes flagged synthetic | **120/120** (100%; 95% CI 96.9–100%) | 60/120 (50.0%) | 88/120 (73.3%) |
| Real speech wrongly flagged | **14/120** (11.7%; 95% CI 7.1–18.6%) | 29/120 (24.2%) | 30/120 (25.0%) |

Per generator, pellav2 caught Chatterbox-Turbo 60/60 and F5-TTS 60/60. False alarms split evenly between clean audiobook speech (LibriSpeech test-clean, 7/60) and noisier speech (test-other, 7/60). ROC AUC was 0.997, so real and fake scores separate well; the misses come from where the 0.5 threshold sits for real speech, which moving the threshold now cannot fix honestly because the test has been seen.

**Channels** (same clips, re-encoded; reported, not part of the bar):

| Condition | Fakes caught | Real wrongly flagged |
|---|---|---|
| Clean | 120/120 | 14/120 (11.7%) |
| MP3 64 kbps | 120/120 | 53/120 (44.2%) |
| 8 kHz telephone (µ-law) | 120/120 | 49/120 (40.8%) |

Compression and telephone channels make real speech look synthetic to this model. That matches the authors' own warning about noisy and recompressed audio, and it is the main limitation for phone-channel use.

**Calibration vs test.** On the separate calibration split (12 speakers), pellav2 flagged 60/60 fakes and 1/60 real clips (1.7%), which met the bar and froze the threshold at the authors' 0.5. The test split (28 other speakers plus test-other) produced 14/120. Twelve speakers was too few to estimate the false-alarm rate reliably; a future protocol should calibrate on more speakers and include compressed audio.

### Design

- **Real speech:** LibriSpeech test-clean and test-other (CC BY 4.0, MD5-verified).
- **Fakes:** Chatterbox-Turbo and F5-TTS v1, generated locally. Each fake clones a LibriSpeech speaker from one 6–11.5 s utterance and reads the transcript of a *different* utterance by the same speaker; that real utterance is in the real set. Speaker and text therefore cannot give the answer away.
- **Splits:** 12 speakers for calibration, 28 for test, no overlap. 60 real + 60 fake for calibration; 120 real + 120 fake for test.
- **Quality gate:** Whisper-base transcribed every clip before any detector scoring. Clips with WER above 0.2 were excluded (38 total, mostly mishearings of names, French phrases, and numbers). Extra candidates were generated in a seeded order, and the first passing clips filled each cell.
- **Frozen:** clip list, hashes, generator versions, and exclusions are in `data/evaluation/modern-manifest.json`, SHA-256 `a464e96a45342e69916b2fd830f38bd59aede4f3c5ff9bbcddcc51875b245834`; a path-free copy is in `modern-set-audit.json`. The receipt is `detector-evaluation-v3.json`. The test split was scored once.

### Limitations

- pellav2's training data is private and may include Chatterbox or F5-TTS output. These are fresh clips from popular open generators, not unseen generator families.
- Two generators, English read speech. ElevenLabs and other commercial voices, voice conversion, and conversational speech were not tested.
- Real speech comes from LibriVox audiobooks (16 kHz, with MP3 history); fakes were generated at 24 kHz. Everything is resampled to 16 kHz before inference.
- Wilson intervals assume independent clips; clips share speakers and texts, so true uncertainty is somewhat larger.

### Reproduce

```sh
.venv/bin/python scripts/setup_pellav2.py
.venv/bin/python scripts/check_pellav2_parity.py
.venv/bin/python scripts/build_modern_set.py plan      # then generate, gate, channels, freeze
.venv/bin/python scripts/evaluate_modern.py calibrate
.venv/bin/python scripts/evaluate_modern.py test
```

Generation needs the two Python 3.11 environments under `data/tts-envs/`; their exact packages are in `scripts/tts/*.lock`.

---

## Legacy: ASVspoof 2019-era protocol (v2)

These results measure 2019-era attacks only. The v2 receipt is `detector-evaluation-v2.json`.


The previous demo supplied a 2.05-second synthetic clip to a reporting policy that abstains below 2.5 seconds. Its raw Wav2Vec2 synthetic score was 0.9959, but the policy overrode the qualified assessment to inconclusive. This was an unsuitable default example, and the small model check also exposed real human false positives.

## Current result

| Stage | Model | Correct binary predictions | Human false positives | Synthetic misses |
|---|---|---:|---:|---:|
| Selection, 20 human + 20 synthetic | Wav2Vec2-XLSR | 27/40 | 13/20 | 0/20 |
| Same selection recordings | AASIST | 36/40 | 4/20 | 0/20 |
| Fresh test, 20 human + 20 synthetic | Selected AASIST | 38/40 | 1/20 | 1/20 |

AASIST is the default. The model choice was frozen before the fresh test was scored; the rule was highest binary balanced accuracy on selection, with a tie favoring the existing model. Thresholds were not fitted to the test. Raw test accuracy is 95%, with a **95% Wilson interval of 83.5%–98.6%**. Test ROC-AUC is 0.9925. These are results on this small ASVspoof sample, not arbitrary upload accuracy or calibrated probabilities.

The separate reliability policy abstained on 15/40 test cases. It made 25 qualified decisions, of which 24 were correct (96%). Reporting only that latter percentage would hide the abstentions. The lab displays both denominators.

## Selection and independence

Sampling seed: `20261020`. Candidate metadata are sorted by official trial ID, permuted with Python's seeded RNG, and selected without reading detector scores. Source groups are unique. Every source group appearing in the official progress subset or the 300-record annotation study is excluded. Decode failures are retained as exclusions, never correct predictions. The current manifest is frozen locally at `data/evaluation/manifest.json` and identified by SHA-256 `970d22877e9fdfbed3edb678a4d3c49eb51f991478870d4dc74f79bfaa7fffc0`.

Per reference class, the first 20 decodable candidates are selection, the next 20 are retired regression cases, and the next 20 are the fresh test. The retired cases were initially tested before a short-tail integration defect was discovered. That defect could repeat a few milliseconds into a full AASIST window, producing spurious disagreements. Version 2 uses a complete context window anchored at the recording end. **The earlier test was retired and 40 fresh source groups were reserved before rescoring.** Both selection models and the unchanged decision policy were rerun, then the chosen model was evaluated on those new test cases. The original receipt remains in `detector-evaluation-v1.json` for audit.

The complete sample has not been audited against all upstream training data. ASVspoof 2021 LA is based on older spoof attacks and channel conditions. Speaker, codec, and attack correlations can remain despite source separation; the simple binomial interval does not model them. This evaluation cannot establish robustness to current voice generators or generalize to all uploaded audio.

## Demonstrations

The demo IDs were frozen before any model comparison: the first two examples of each reference class lasting 2.5–13 seconds. They are 2.90 and 4.46 seconds (human), and 4.04 and 4.166 seconds (synthetic). No prediction or confidence filter was used. With the corrected default, all four match the reference and pass the reliability policy. The largest score shifts under the three configured sensitivity probes are approximately 0.000153, 0.002471, 0.000059, and 0.000000. See `reference-demonstrations.json` for exact records. These four visible examples are demonstrations, not additional held-out performance evidence.

## Reproduce

```sh
.venv/bin/python scripts/setup_detector.py
.venv/bin/python scripts/setup_aasist.py
.venv/bin/python scripts/prepare_evaluation.py
.venv/bin/python scripts/evaluate_detectors.py
.venv/bin/python scripts/check_detector.py --output data/analyses/reference_check.json
```

The preparation script requires the existing registered ASVspoof database and verified official archive. On a new database, internal IDs and local paths may differ; official trial IDs, source groups, source hashes, and the seeded partition define the same recordings. Model revisions and weight/config/source hashes are stored in `detector-evaluation.json`. Inference reports retain every score, both logits, source checksum, window, preprocessing rule, and package version. A later model or preprocessing change requires a new protocol version and fresh confirmation data.

The exact 80-case partition was regenerated from the seed and original metadata and matched the executed manifest. The portable sampling audit, including all decode exclusions and a clarification of the legacy sampling-description field, is in `evaluation-sampling-audit.json`.
