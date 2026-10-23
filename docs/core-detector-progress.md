# Core AI-voice detector progress

The product stays focused on upload → automatic assessment → clear result.
This work evaluates the actual AI-voice detector, not speech activity or acoustic
markers. Frozen production weights and thresholds are unchanged.

## Modern-generator screening

A preselected 45-clip public development screen contains 20 publisher-labeled
English human recordings and 5 clips each from Chroma, GLM-TTS, Qwen3-TTS,
VibeVoice and VoxCPM. Paths were ordered by SHA-256 of a fixed seed string plus
publisher path before inference. Both fixed/random TTS source folders were
eligible. The 45 files total 12,192,707 bytes; the full 23 GB dataset was not
acquired. Source attribution: Aastha Sharma and Guangjing Wang,
[VoxENES 2026](https://arxiv.org/abs/2607.11706),
[public dataset](https://www.kaggle.com/datasets/interspeech2712/voxenes-2026),
version 1, publisher-declared CC BY 4.0. The 132 cached catalog pages are a partial
listing, not a complete inventory; all five chosen English original-TTS folders
were present. Voice conversion and Spanish are not covered here.

On the original-condition center crops, the current frozen fusion flagged
**25/25 synthetic clips**, with **0/20 raw false alerts on human clips**.
23 synthetic clips received a definitive synthetic indication, 16 humans a human
indication, and 6 clips were inconclusive. There were no wrong definitive results
in this small clean screen. See the [initial receipt](core-modern-screen-v1.json).
This is new evidence about the existing detector, not an accuracy improvement
from a model change.

### Interpretation limits

All clips conservatively share one unresolved speaker/source identity group.
Publisher labels are used; upstream detector exposure and speaker overlap with
local history remain unknown. These are development results, not an independent
locked test or a population estimate. Five examples per generator cannot
establish robust performance against a whole generator family. Zero false alerts
in 20 human clips does not establish a low population false-positive rate.

Native downloaded files span 2.315–40 seconds and 16/24/44.1 kHz, despite the paper
and dataset description stating standardized four-second 16 kHz audio. The runner
therefore records native headers, mixes/resamples through the existing mono16
path, and selects at most four seconds at the center without adding padding.
Model-specific short-input handling remains unchanged. This policy was fixed
before scoring and applied to both classes. It differs from any unverified
publisher preprocessing implementation; our counts do not reproduce or directly
rank against the paper's EER table. The paper itself reports substantial temporal
generalization failures across the detectors it evaluated; it did not evaluate
our local fusion.

Core screening applies the existing quality and uncertainty-band decisions but
does not run the product's extra perturbation-stability check or score complete
long recordings. Therefore, it is not end-to-end product accuracy.

## One rejected core-model candidate

A fixed candidate clipped both component logits to ±5 before fitting the same
logistic fusion, using the historical L2, speaker folds and threshold rule.
The motivation was to prevent a saturated component from dominating. It failed:

| Historical development condition | Baseline human false alerts | Candidate human false alerts |
| --- | ---: | ---: |
| Clean | 1/180 | 7/180 |
| MP3 | 5/180 | 40/180 |
| Telephone | 7/180 | 35/180 |
| Noise 20 dB | 11/180 | 42/180 |

The candidate was rejected. No production weights or threshold were changed.
The [experiment receipt](core-bounded-fusion-rejected-v1.json) includes complete
out-of-fold rows and source hashes. It reuses historical development and its
pooled out-of-fold threshold rule; this is not untouched calibration or a nested
independent model-selection study. No parameters were adjusted to rescue it.

## Reproduce

```sh
.venv/bin/python scripts/acquire_core_screen.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/screen_core_detector.py \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --output /tmp/new-core-screen.json
# Fixed paired original/gain/noise/bandwidth/Vorbis conditions on BOTH classes:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/screen_core_detector.py --stress \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --output /tmp/new-core-stress.json
.venv/bin/python scripts/evaluate_bounded_fusion.py \
  --output /tmp/new-rejected-candidate.json
```

Use new output filenames. Acquisition verifies observed SHA-256 and source byte
sizes pinned in [the acquisition receipt](core-modern-acquisition-v1.json);
these SHA-256 values were computed locally, not supplied by the publisher.
Downloads are capped at 4 MiB per file and 16 MiB total, with approved HTTPS
redirect hosts. No paid service, cloud inference or additional model download is
required. Scored input identities receive durable content-addressed prior-use
receipts so they cannot silently become fresh locked data later.

## Paired stress results and a shipped reliability correction

The same 45 originals were tested in five fixed conditions, applied to both
classes: original, −12 dB gain, constructed white noise at a 20 dB
whole-mixture/added-noise RMS ratio, 8 kHz bandwidth round trip, and Vorbis.
These are **225 correlated case rows**, not 225 independent originals.
The noise ratio is not clean-speech SNR and transformed audibility is not
independently relabeled. Full counts are in the
[pre-guard stress receipt](core-modern-stress-v1.json).

| Condition | Raw synthetic detections | Raw human false alerts | Wrong definitive assessments before guard |
| --- | ---: | ---: | ---: |
| Original | 25/25 | 0/20 | 0 |
| Gain −12 dB | 25/25 | 0/20 | 0 |
| Added noise | 19/25 | 1/20 | 2 |
| Bandwidth 8 kHz | 23/25 | 0/20 | 0 |
| Vorbis | 25/25 | 1/20 | 0 |

Two noisy Qwen3 examples received definitive human indications, despite PellaV2
flagging synthetic scores 0.934 and 0.969 and AntiDeepfake strongly favoring human.
Both mistakes also occurred in full product reruns with stability checks enabled;
see the [targeted pre-fix followup](core-failure-followup-v1.json). This confirms
a product failure, rather than merely a screening difference.

Policy 1.1 now requires corroboration for a **human** indication: both component
aggregate class votes must favor human at their default 0.5 thresholds. Component
aggregation uses the same window weights as the file score. A conflict retains
the raw fused score and binary decision for technical evidence, but the user
assessment becomes **Inconclusive**, with a plain-language reason. Synthetic
indications still follow the frozen fusion and existing quality/uncertainty/
stability checks. No weights, learned threshold, model score or UI control changed.
This is an engineering abstention guard, not a newly calibrated detector.

A broader initial rule deferred both human and synthetic indications whenever
components disagreed. It was rejected because definitive coverage fell from
186/225 to 127/225 and synthetic indications fell from 104 to 58. Its
[rejected replay](core-component-guard-broad-rejected-v1.json) is preserved.
The final guard targets false reassurance while retaining AI indications:

| Development observation | Before | Final human-corroboration guard |
| --- | ---: | ---: |
| Wrong definitive assessments | 2 | 0 |
| Definitive synthetic indications | 104 | 104 |
| Definitive human indications | 82 | 69 |
| Inconclusive | 39 | 52 |
| Definitive coverage | 186/225 (82.7%) | 173/225 (76.9%) |

The [final replay and product verification](core-component-guard-v1.json) binds
inputs and source hashes. Both known noisy-Qwen product cases were rerun with
real models after the correction and became inconclusive. Other cases replay
existing raw scores with verified reconstructed inputs, not fresh full product
inference. The guard was designed **after** observing these failures and its
coverage was inspected on the same development cases: these are post-hoc
engineering results, not independent evidence of improved population accuracy.
Raw noisy-synthetic detection remains 19/25; the guard does not turn misses into
AI detections. Strengthening that sensitivity remains core-model work.

Reproduce the final guard replay (new output path):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/replay_component_guard.py --verify-product \
  --screen docs/core-modern-stress-v1.json \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --output /tmp/new-human-corroboration-replay.json
```

Rerunning the screening command with current code uses policy 1.1. Earlier
receipts intentionally retain their original policy and source hashes.
The next detector milestone is better sensitivity to noisy modern synthesis,
followed by larger source-diverse calibration and untouched external evaluation.
No manual review step has been added to the product workflow.

## Subsequent noisy-voice experiments

Three fixed preprocessing candidates were tested against both classes in clean
and constructed-noise conditions. None safely improved the target sensitivity;
all were rejected. The neural enhancer recovered raw noisy AI detections but
falsely flagged every noisy genuine clip. See the
[noise improvement study](core-noise-improvement.md). The live detector remains
unchanged; the next core candidate must address noise robustness in the detector
rather than assume cleaner audio preserves forensic evidence.

## Alternative classifier comparison

The already-installed third classifier was also rejected: 11/25 clean and
0/25 noisy synthetic detections, with 5/20 clean human false alerts. See the
[paired classifier comparison](core-classifier-comparison.md). A newer WavLM
candidate was inspected but deferred before large weight acquisition; it has
not been evaluated or enabled. The live model remains unchanged.

## Compact noise-augmented model

RawNetLite was acquired under a 1.1 MB cap and tested on all 225 existing
development rows. It was fast (p50 36 ms) but detected only 6/25 clean and
6/25 noisy synthetic clips; noisy human false alerts were 4/20. It is rejected
and is not available in the app. See the [decision and reproduction details](core-rawnetlite-decision.md).

## Complementary model investigation

See the [capability update](detector-capability-update.md): larger models, full
1,280-row regression, GPU parity and a fixed partial-synthesis challenge. The
raw OR candidate recovers one noisy v4 miss without adding false alerts, but
requires independent calibration and validation before product promotion.

## Calibration probe

A retrospective connected-group split calibrated the complementary detector
without using evaluation scores to choose its threshold. The evaluation noise
cell catches52/53 synthetic recordings with1/100 genuine alerts. This threshold
is not promoted; see [calibration details](complementary-calibration.md).

## Unscored external acquisition

An official ASVspoof5 pilot now contains40 new-to-project recordings from40
declared speakers, with balanced reserved partitions and no detector scores.
See [acquisition and qualification limits](asvspoof5-pilot.md). It is not yet a
qualified independent calibration or locked evaluation corpus.
