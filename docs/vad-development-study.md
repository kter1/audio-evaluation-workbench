# Annotated VAD development study

Thirty publisher-annotated TEN VAD clips provide 13,104 common 20 ms frames
(262.08 seconds). All clips share one conservative source group because original
speaker/source identities are unresolved. These are exploratory development
results, not independent population estimates or a production promotion.

| Method | Speech recall | Nonspeech false-positive rate | Frame F1 |
| --- | ---: | ---: | ---: |
| Energy activity diagnostic | 99.06% | 88.95% | 86.79% |
| WebRTC, mode 2 | 94.10% | 50.22% | 89.37% |
| Silero, raw probability ≥ 0.5 | 93.79% | 18.77% | 93.81% |

The common reference contains 9,864 speech and 3,240 nonspeech frames. Silero
reduces false speech detections with slightly lower speech recall than WebRTC on
these clips. No threshold was tuned. No confidence interval implying independent
speakers is reported. Publisher labels have not received independent review;
training exposure, source provenance and domain coverage remain unresolved.

Silero uses pinned CPU TorchScript inference on 512-sample (32 ms) chunks at
16 kHz. Native predictions are projected onto the common 20 ms frame centers;
the final partial model chunk is zero padded. This measures raw frame decisions,
not Silero's official timestamp postprocessing. There is no tuned hangover,
padding or minimum speech duration. WebRTC uses 20 ms PCM frames and mode 2.

## Reproduction and provenance

[Acquisition specification](research-acquisition-v1.json) pins revisions, sizes,
Git blob identities and the ASV archive publisher checksum. Acquisition scripts
bound downloads, validate redirects and verify bytes before publishing files.
Raw inputs stay in ignored local data directories. TEN's license contains
additional conditions; this acquisition establishes no commercial clearance.
The Silero model has its pinned license alongside the local model.

```sh
.venv/bin/python scripts/acquire_research_data.py ten-vad-development
.venv/bin/python scripts/setup_research_vad.py
.venv/bin/python scripts/acquire_research_data.py silero-research-model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=data/research-tools/webrtcvad-2.0.14.post1/site-packages \
.venv/bin/python scripts/benchmark_methods.py \
  --manifest data/evaluation/ten-vad-development-v1.json \
  --silero-model data/research-tools/silero/066409f3d9db45a30d21dfcbf9147826ffdd7052/silero_vad.jit \
  --output /tmp/new-vad-comparison.json
```

Output paths must be new. Supplied-audio runs also save content-addressed prior-use
receipts, preventing these consumed development identities from silently becoming
fresh evaluation data. The optional WebRTC wheel is isolated from the main venv.
See [runtime provenance](research-vad-runtime-v1.json), the initial
[two-method run](vad-development-v1.json), and the final
[three-method run](vad-development-v2.json) for measured counts and source hashes.

## ASVspoof 5 metadata audit

Only the 20,669,392-byte protocols archive was acquired; no ASVspoof 5 audio was
downloaded. Its compressed publisher MD5, three member checksums and row counts
were verified without extracting files. Train/dev/evaluation have respectively
182,357 / 140,950 / 680,774 rows. Evaluation has 737 declared connected genuine
speaker groups and 367 synthetic speaker groups; codec-parent references resolve.
These are metadata relationships, not verified physical speaker independence.
Attack IDs do not establish generator-family independence or training exclusion.
The protocols supply no reviewed speech/noise/edit interval annotations.
See the [audit receipt](asvspoof5-metadata-audit-v1.json).

Next work is independently reviewed, provenance-backed development material with
broader domains, plus separately reserved calibration and locked-test data.
Candidate promotion requires the frozen evaluation contract. Production scoring
and anomaly rules remain unchanged by this study.

## Independent-review preparation

A local review packet can be prepared without exposing publisher labels or model
predictions to the reviewer:

```sh
.venv/bin/python scripts/prepare_speech_review.py \
  --manifest data/evaluation/ten-vad-development-v1.json \
  --output data/evaluation/review-packets/ten-vad-v1
```

The destination must not exist. All manifest cases are included, with random
ordering, neutral clip IDs, verified original audio bytes, durations and empty
annotation fields. Only `reviewer/` is intended for a reviewer; keep
`coordinator-only/` private. Source license restrictions still apply before
sharing. Audio content may itself reveal identity; this is blinding of supplied
metadata and predictions, not a guarantee of anonymity.

The local TEN packet contains 30 pending clips. No reviewer has been contacted
and no independent labels are claimed. A reviewer must listen to complete clips,
record speech intervals and ambiguity notes, and identify themselves. Independent
reviewer assignment, second-review/adjudication policy and completed-label
validation remain required before computing label agreement. Publisher annotations
and model results must remain unavailable during initial annotation. Reviewed
labels still belong to development; they cannot make these clips a fresh test set.

## Completed-review validation and agreement

After two reviewers return complete annotation files, run:

```sh
.venv/bin/python scripts/compare_speech_reviews.py \
  --packet data/evaluation/review-packets/ten-vad-v1 \
  --left /path/to/reviewer-one.json --right /path/to/reviewer-two.json \
  --output /tmp/new-review-agreement.json
```

Every packet case must be present exactly once, with unchanged clip identity,
duration and verified audio bytes. Completion requires a reviewer identifier,
timezone-bearing completion timestamp, reviewed status, textual notes and finite,
in-range, nonoverlapping speech intervals. Empty intervals mean reviewed silence;
null remains unreviewed and blocks comparison. Different reviewer identifiers are
required but do not prove that people worked independently.

The report includes paired 20 ms frame counts, observed agreement, Cohen's kappa,
per-clip disagreement duration and a queue of clips requiring adjudication. Kappa
is undefined when its chance-agreement denominator is zero. Frame centers use
[start,end) membership; incomplete final frames are excluded. Any interval
mismatch or reviewer note also triggers adjudication, including subframe boundary
differences. No majority-vote labels are created automatically. Counts describe
agreement rather than accuracy, and no confidence interval implying independent
speakers is produced. Review file hashes and analysis source hashes preserve the
comparison inputs and implementation. This is a review-quality diagnostic, not a
promotion gate or evidence of completed review.

The real 30-clip packet still contains pending labels. Validation has been tested
using constructed reviewer files; no human reviewer results are fabricated.

## Explicit adjudication and development export

Prepare decisions from the two original completed reviews, then have an
adjudicator listen to disputed clips without model predictions:

```sh
.venv/bin/python scripts/adjudicate_speech_reviews.py prepare \
  --packet data/evaluation/review-packets/ten-vad-v1 \
  --left /path/to/reviewer-one.json --right /path/to/reviewer-two.json \
  --output /tmp/new-pending-decisions.json
```

Each disputed clip includes both reviewers' intervals and notes. Final decisions
require resolved status, explicit speech intervals (including [] for nonspeech)
and a nonempty rationale. Subframe differences and notes remain disputes. The
adjudicator must provide an identifier different from both initial reviewers for
disputed cases, plus a timezone-bearing completion time. These identifiers do not
prove real-world independence. No-dispute packets still require coordinator
signoff; exact consensus intervals are retained.

```sh
.venv/bin/python scripts/adjudicate_speech_reviews.py export \
  --packet data/evaluation/review-packets/ten-vad-v1 \
  --left /path/to/reviewer-one.json --right /path/to/reviewer-two.json \
  --decisions /path/to/completed-decisions.json \
  --manifest data/evaluation/ten-vad-development-v1.json \
  --output /tmp/new-adjudicated-development.json
```

Export requires every disputed case exactly once, unchanged review context,
matching review/mapping hashes and the original manifest hash. It verifies packet
audio again, checks interval bounds/overlap, retains original manifest source
identities and records review/adjudication evidence hashes. Publisher annotation
provenance is retained separately. Original reviews, source manifest and existing
outputs are not overwritten. The exported manifest works with the existing
`benchmark_methods.py --manifest` development runner, which records subsequent
use; no partition is promoted to calibration or locked test. No corpus
qualification, independent-review claim or production change follows from export.
Only constructed review fixtures have exercised successful export so far.

## Reusable-model verification and resource measurements

A subsequent [comparison](vad-development-v3.json) reuses one Silero model per
run with recording state reset. All 30 clips retain identical speech outputs and
raw scores relative to v2. The [local runtime study](vad-runtime-study.md) reports
cold setup, warm latency by duration, real-time factor and whole-process memory.
These resource observations do not substitute for independent label review.

## Automated robustness extension

The [automated stress study](vad-stress-study.md) measures fixed-method behavior
under gain, constructed noise and exact silence-prefix transformations. It runs
without manual review, retains original grouping and inherited-label limitations,
and reports each condition separately rather than inflating independent support.
