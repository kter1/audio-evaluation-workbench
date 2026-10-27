# Capability investigation: complementary detector found

The current simple product and frozen fusion weights remain in place. Its negative
UI result now says **No clear AI signal**, explaining that short AI inserts can
be missed. This is an interpretation correction, not an accuracy gain.

## Actual model results

The current public Pella release is a different W2v-BERT model from the installed
XLS-R version. Its authenticated checkpoint contains 12,154 saved NaNs in one
parameter tensor. It was rejected before accuracy evaluation. No weights were
repaired. See [checkpoint health](pella-current-checkpoint-health-v1.json).

The second candidate, WavLM-large+AASIST, passed checksum, finite-weight and
strict inference-state checks. Four unused training projection tensors were
individually authenticated and excluded, matching the published inference
architecture. Native preprocessing and score direction were tested. Its weights
are CC-BY-NC-4.0; research use does not establish commercial redistribution rights.

On the repeated 45-source modern screen, standalone WavLM detects 21/25 clean
and 18/25 noisy synthetic clips, versus 25/25 and 19/25 for the current fusion.
It produces zero raw false alerts on the 100 correlated genuine variants
(20 genuine originals). It rescues two noisy misses and one bandwidth miss.
It is complementary, not a better standalone replacement.
[Full modern screen](core-wavlm-aasist-screen-v1.json).

## Broader paired regression completed

All 320 v4 originals, four stored conditions, 1,280 rows and 73 connected groups
were scored using the candidate's native five-second anchored windows. Historical
fusion uses four-second windows: this is an identical-file system comparison,
not identical model input. No threshold fitting occurred.

A raw binary OR rule was frozen after modern development results and before the
broader results: preserve existing fusion flags or add WavLM flags at its native
0.5 threshold. This post-hoc development rule is not the product decision policy.

| Condition | Current AI detected | Combined AI detected | Current / combined human false alerts |
| --- | ---: | ---: | ---: |
| Clean | 120/120 | 120/120 | 0/200 / 0/200 |
| MP3 | 120/120 | 120/120 | 3/200 / 3/200 |
| Phone | 120/120 | 120/120 | 5/200 / 5/200 |
| Added noise | 119/120 | 120/120 | 10/200 / 10/200 |

Standalone WavLM had zero human raw false alerts in each condition, and detected
120/120 synthetic clips in clean/MP3/phone and 114/120 under added noise.
The OR combination passes the frozen no-additional-false-alert regression gate,
but retains existing false alerts. These are raw flags, not definitive product
assessments. [Paired decision](three-model-regression-decision-v1.json),
[complete regression](core-wavlm-aasist-v4-mps-regression-v1.json).

The Apple GPU passed all ten fixed CPU parity cases, maximum probability error
0.00000507 and native-logit error 0.0000358. The duplicate CPU regression was
stopped after at least 440 rows; only the complete GPU run supplies reported
regression results. [Device parity](wavlm-mps-parity-v1.json).

## Short inserts remain a gap

Four genuine 12-second contexts received fixed 1/2/4-second generated inserts,
with pristine and same-speaker genuine-splice controls: 28 correlated cases.
The current fusion indicated human for all four one-second insert cases and
abstained on all two/four-second cases. WavLM indicated human on seven partial
cases. Neither falsely indicated AI on the 16 genuine controls. The one-second
excerpts contain substantial energy; speech content was not independently reviewed.
These constructed results do not establish natural mixed-audio accuracy or
localization. [Fusion challenge](core-partial-fusion-v1.json),
[WavLM challenge](core-partial-wavlm-v1.json).

## Release decision

Retain the current detector while developing calibrated use of the complementary
model. Spent regression data does not justify promoting it to production. Source-diverse calibration and untouched independent evaluation,
including brief inserts and real noisy recordings, remain necessary. Large
downloads were bounded, one-time research acquisitions; opening the app still
never downloads models. All processing stayed local.

Checks: 363 Python tests and 6 frontend tests passed; compilation, JavaScript
syntax, diff checks and recorded regression code hashes passed.
