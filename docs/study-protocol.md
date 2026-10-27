# Reproducible study protocol

## Before listening

The optional Workspace warm-up uses reference-labeled progress recordings whose source groups are excluded from all study assignments. It offers immediate learning feedback and creates no formal review records. Do not count practice impressions as study work.

Keep a fresh research database separate from the demo. Register the unmodified official metadata and published LFCC-GMM scores. Their SHA-256 fingerprints are recorded. `prepare-study` freezes the threshold before selecting recordings; rerunning does not resample or recalibrate a frozen study.

Threshold calibration uses all official `progress` rows, requiring complete scores and mappings for that subset. Higher score means human; `score >= threshold` is human. The empirical, tie-aware operating point minimizes the absolute difference between genuine rejection and spoof acceptance; ties choose the lower threshold. This is a study operating point, not a claim about a production detector or the challenge's official threshold.

Sampling excludes unmapped evaluation recordings and every source group represented in progress. It chooses one recording per remaining source group with a fixed seed, shuffles within balanced classes, and forms the four batches. Media is decoded and hashed before assignment. Broken media is excluded deterministically and recorded in the private manifest. Only original, readable files can be submitted.

Study phases: `calibration` (20), `main-01` (80), `main-02` (100), `validation` (100). Calibration is excluded from performance summaries. Inspect and revise guidelines after calibration, before main work starts. Primary batch manifests are frozen in SQLite; separate workspaces produce different random opaque assignment IDs but the same seeded source selection.

## Daily review

Use comfortable playback volume and take breaks; do not infer productivity or quality percentages from time spent. Save drafts as needed. On submit, the original judgment becomes immutable. Blind exports expose assignment status only; submitted answers are hidden until reveal, including during the repeat waiting period.

For `main-01`, complete all 80 primary assignments, then schedule exactly 20 repeats. Each opens 48 hours after its own original submission; no interface bypass is provided. Repeat aliases do not expose their original assignment identity. Finish them before reveal. Practice uses a smaller 28-recording sample; select two repeats for its first main batch.

## Independent quality check

Quality Review now provides **Get an independent review**: enter the reviewer's name and recording count, prepare and download a blind package, then import their returned results. Existing packages remain available for re-download without assigning duplicate work. The CLI equivalents below remain supported.

Ask another person to participate yourself. This software never contacts anyone. Use the same guide and keep communication free of your judgments or reference labels. Target **20 main-01 + 30 main-02 = 50 main** reviews, followed by **30 fresh validation** reviews.

Before revealing each relevant batch:

```sh
.venv/bin/python -m lab --db data/study.sqlite export-package --reviewer second-reviewer --phase main-01 --count 20 --output exports/main01_blind.zip
```

Give them the repository and blind ZIP, which contains neutral WAV media, assignment IDs, the guideline, and generic source attribution. It contains no source filenames, per-recording metadata, reference labels, scores, or earlier judgments. On their machine:

```sh
# Install with sh run-legacy.sh once, then stop the practice app.
.venv/bin/python -m lab --db data/second-main01.sqlite load-package --file /absolute/path/to/main01_blind.zip
AUDIO_LAB_DB=data/second-main01.sqlite sh run-legacy.sh
# After all assigned judgments are submitted:
.venv/bin/python -m lab --db data/second-main01.sqlite export-results --output exports/main01_results.json
```

Use a new reviewer database for each package. Return only the results JSON. Import it through Quality Review or:

```sh
.venv/bin/python -m lab --db data/study.sqlite import-results --file /absolute/path/to/main01_results.json
```

Package assignment identity, reviewer identity, annotations, timestamps, and checksums are validated. Duplicate IDs, missing reviews, conflicting returns, and mismatched packages are rejected atomically. An identical resend is a no-op. Creating a package creates pending independent assignments, blocking reveal until they return; create packages only for reviewers who have agreed to participate. There is no silent cancellation that could conceal missing work.

Without an independent reviewer, continue the study and report independent agreement as **not measured**. Your own repeats are not independent review.

## Revealing, correcting, and adjudicating

Reveal only after every assigned primary, repeat, and independent judgment in the batch is submitted. `main-01` additionally requires scheduled repeats. Once revealed, the phase cannot be reused for blind work. Use Quality Review to inspect perceptual judgment, dataset reference, and published detector prediction separately.

Corrections append revisions; adjudications append an explicit decision and rationale. Neither changes original metrics. Compare original judgments before discussing disagreements. You may disagree about quality descriptions while both following the guide; do not manufacture consensus.

## Revised guideline and fresh recordings

Copy the guide, revise it based on documented ambiguity, then register a new version before its batch has any draft or submitted records:

```sh
.venv/bin/python -m lab --db data/study.sqlite guideline --version 2.0 --file /absolute/path/to/revised-guide.md --phase validation
```

Versions are immutable. Reassigning a started phase to another guideline is rejected. Report results by phase and guideline; different recordings prevent a causal claim that a revision improved accuracy. If the guide does not change, keep 1.0 and say so.

## Research handoff

Use the case filters in Quality Review to find detector mistakes, reviewer disagreements, escalations, and uncertain judgments. Calibration comparisons are explicitly labeled learning feedback; main-study metrics continue to exclude them. Findings provides a research ZIP with the report and full supporting records.

Write five case studies in Quality Review. Include exact observations and timestamps, distinguish hypotheses from established causes, and propose a concrete next check. Include an unresolved case if one exists; do not force an error category the data did not produce.

Export original annotations, revision history, metrics, and the findings report. The printable HTML has a summary section and a case-study/limitations section. Keep complete notes in the companion exports; no report should imply a completed study while cases or reviews are pending.

Human metrics: uncertainty rate across reviewed cases and accuracy only among decided cases, always displayed together. Detector metrics: genuine rejected / genuine total; spoof accepted / spoof total. Missing scores are counted separately. Agreement includes `uncertain` as a category and only compares the same guideline version. Kappa is unavailable if expected agreement equals one. Small, balanced benchmark slices cannot establish deployment reliability, fraud prevalence, or causation.
