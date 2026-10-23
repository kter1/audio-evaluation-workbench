# SignalTrace methods audit — baseline 1.0

This is the first implementation milestone of the methods improvement plan.
It establishes the baseline and a prospective evaluation contract. It does not
establish world-class performance or replace the existing detector.

## Reproduce

From the repository root, using the existing environment:

```sh
.venv/bin/python scripts/audit_baseline.py
# Optional: save to a NEW file; an existing output is never overwritten.
.venv/bin/python scripts/audit_baseline.py --output /tmp/new-baseline-audit.json
```

The command requires the local development/evaluation manifests, feature cache,
source audio and archived v4 reports. It performs no downloads or neural
inference. Exit 0 means integrity and archived replay passed; exit 1 means
completed integrity checks found errors; exit 2 means inputs are missing,
malformed, or the audit could not complete. Statistical limitations and missed
model targets are reported separately from integrity. A clean checkout without
the ignored `data/` assets cannot reproduce the audit from source alone.

The generated [audit receipt](baseline-audit-v1.json) records input hashes,
audit-code hash, environment, per-condition counts and speaker-bootstrap
intervals. The original frozen evaluation and fusion files are unchanged.

## Verified findings

- All 2,120 referenced development/evaluation audio files matched their hashes.
- Both manifest self-digests, fusion identity, model identities, threshold,
  development feature-cache identity and coverage passed the checks.
- All 1,280 archived recording-condition reports passed their content hashes.
  Replaying their logit-to-score conversion and weighted aggregation reproduced
  receipt scores with maximum absolute difference **0.0**.
- Development versus evaluation shared **zero** speaker IDs, audio byte hashes,
  source/cloning-reference utterance IDs, or normalized full texts. Normalization
  compares case and whitespace only; this is not a perceptual-duplicate audit.
- The evaluation has **320 source recordings, 73 speakers and four conditions**.
  The 1,280 variants are correlated, and some source texts are shared across
  human recordings and cloned renditions within the evaluation set.

| Condition | Human falsely flagged | Synthetic detected |
| --- | ---: | ---: |
| Clean | 0 / 200 | 120 / 120 |
| MP3 | 3 / 200 | 120 / 120 |
| Telephone simulation | 5 / 200 | 120 / 120 |
| Added white noise, 20 dB SNR | 10 / 200 | 119 / 120 |

These counts use the frozen binary threshold, including recordings whose
assessment abstains. The receipt also reports abstention and accuracy among
decided recordings. Neither clean accuracy nor aggregate accuracy establishes
reliability in unseen conditions.

The 2,000-draw speaker bootstrap gives an approximate **1.93–9.50%** interval
for the noisy-human false-positive rate. The clean zero-error bootstrap interval
collapses to zero and must not be interpreted as zero population risk. Bootstrap
intervals describe these sampled speakers; they do not cover unknown generators
or recording domains. The prospective contract requires an additional rare-error
safeguard and substantially more independent groups.

## Fresh inference spot check

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/check_receipt_regression.py --per-channel 1
```

Executed locally with the existing checkpoints and the script's seeded sampling:
**4 recording-condition versions, 0 binary verdict flips**, largest absolute score
difference **1.35e-03**, largest logit difference **0.103**. This checks one clip
per condition, not the full neural baseline. The existing script fails on verdict
flips; it does not impose a numerical score-drift tolerance. These observed
differences therefore establish sampled verdict stability, not exact numerical
reproduction of current inference. Its sampling and original outputs remain
available in the existing script and frozen receipt.

Archived per-file runtime median was **1.27 s**, p95 **2.71 s**, and summed runtime
divided by summed audio duration **0.215**. These are historical timings,
including per-report analysis work but not a separately measured cold model load;
they are not current-machine performance measurements. Peak memory was not
recorded. The new resource gates remain unverified.

## Method weaknesses and supported interpretations

1. **Development gate was missed.** The frozen base fusion's cross-validation
   flagged 11/180 noisy human versions (6.11%), above its 5% target. The existing
   specification explicitly records that the base was frozen despite that miss.
   The later evaluation passed its point-estimate targets. Preserve both facts;
   a future candidate cannot waive a failed promotion gate under the new contract.
2. **The published test is spent.** Existing results have been inspected and
   discussed. Preserve this corpus for regression and development only. A fresh
   locked corpus has not been acquired, partitioned or evaluated.
3. **Scope is narrow.** English read speech, three synthesis systems and simulated
   channels do not establish conversational, multilingual, device, replay or
   unseen-generator-family performance. XTTS is held out from local fusion fitting;
   this does not prove it was absent from upstream detector training.
4. **Acoustic markers are heuristics.** Current rules use 20 ms levels, fixed
   quiet/saturation thresholds, 120 ms level comparisons and five-band spectral
   differences. Marker caps and strongest-event selection limit recall. No
   natural-recording event precision/recall study establishes their reliability.
5. **Representation matters.** RMS, peak, spectral descriptors and event rules
   mostly use resampled arithmetic-mean mono at 16 kHz. Original-channel clipping,
   channel RMS and stereo correlation are measured separately. Mono mixing can
   cancel opposite-phase content; resampling discards high-frequency evidence.
   The event saturation rule uses 0.995 on derived mono, while the original-channel
   clipping metric uses 0.999. Neither should be mistaken for the other.
6. **Quality activity is not speech activity.** The -50 dBFS activity rule detects
   energy. Music and noise can pass it. The noise index measures frame-level
   dynamic spread; it is not a validated SNR estimator.
7. **Window scores are coarse classification.** Four-second windows, including
   overlapping end-anchored tails, are averaged equally. This can dilute a short
   synthetic passage and gives overlapping tail samples extra representation.
   No validated edit boundary or generator attribution follows from these scores.
8. **Confidence is uncalibrated.** The logit abstention band and component score
   difference rule are engineering policies. Component magnitudes have different
   scales. Perturbing one energetic window establishes limited sensitivity,
   not correctness or whole-recording robustness.
9. **An audit-exposed regression was corrected.** A legacy test still rejected
   121-second recordings after the supported limit became 600 seconds. It now
   verifies 121 seconds is accepted and 601 seconds rejected. The stale 50 MiB
   decoder error message now matches the actual 256 MiB limit. No scoring,
   thresholds, or source preprocessing changed.

## Prospective protocol and next work

[Evaluation contract v1](evaluation-contract-v1.json) fixes numerical targets
before candidate tuning: calibration-selected operating points, independent-group
sample minimums, confidence safeguards, condition-specific regression limits,
calibration diagnostics, separate localization targets and CPU/memory budgets.
These are acceptance requirements, not achieved measurements or an implemented
automatic promotion service. The contract deliberately prevents an internal
benchmark pass alone from authorizing a world-class claim.

Next implementation work is corpus inventory/acquisition planning and validated
measurement corrections, followed by at most three detector candidates. Before
locking a new corpus, construct connected source/speaker/reference groups and
audit perceptual duplicates, provenance and training-contamination risks. The
current audit covers the v4 fusion development/evaluation boundary, not every
historical dataset. No large downloads, training or paid compute were performed.

Methodological reference: the [ASVspoof 5 evaluation](https://arxiv.org/abs/2601.03944)
motivates explicit testing of out-of-domain performance. It supplies no validation
for SignalTrace itself.
