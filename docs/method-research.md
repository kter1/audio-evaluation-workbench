# Development methods and corpus qualification

Milestone 3 adds an executable qualification gate, development method comparisons,
and durable prior-use tracking. No experimental method is connected to studio
findings, neural inputs or classifier scores. The evaluation contract is unchanged.

## Run the controlled comparison

```sh
.venv/bin/python scripts/benchmark_methods.py --fixtures
# Optional NEW output path; existing evidence is not overwritten.
.venv/bin/python scripts/benchmark_methods.py --fixtures --output /tmp/method-fixtures.json
```

[Fixture receipt](method-benchmark-fixtures-v1.json) contains **22 non-speech
signals in six connected source groups**: stationary Gaussian noise, 6/12/18 dB
gain steps, gradual gain ramps, silence and a steady tone. Gain variants share
their underlying noise source and remain in one bootstrap group. The runner
additionally connects identical decoded sample hashes even when metadata claims
different source groups.

| Level-change method | Matched events | False events | Missed events | Event F1 |
| --- | ---: | ---: | ---: | ---: |
| Frozen production level rule | 8 | 0 | 4 | 0.80 |
| Experimental adaptive rule | 12 | 0 | 0 | 1.00 |

This deliberately small comparison exercises known gain changes. It does not
establish superiority on speech, music, edits or real recording conditions.
**Promotion is explicitly ineligible.** There are no annotated speech-positive
frames in these fixtures, so speech recall remains undefined.

## Compared methods

- **Frozen level rule:** executes the existing `acoustic-1.0` implementation and
  extracts its level-change markers, retaining its thresholds and selection caps.
- **Adaptive level rule:** compares mean power across adjacent 120 ms windows at
  a 20 ms hop. Its threshold is the greater of 6 dB and six times a robust local
  spread estimate: 1.4826 times median absolute deviation of adjacent frame-level
  differences within a two-second context on either side. Both sides must exceed
  -50 dBFS. Strongest-first suppression separates candidates by 300 ms. Returned
  intervals express window support, not exact edit boundaries. This is a new
  experimental heuristic, not a published or independently validated detector.
- **Noise baselines:** a global lower-decile frame-power statistic and the median
  of lower-decile statistics in two-second blocks. Both can mistake continuous
  speech or changing background conditions for noise.
- **Assumption-specific noise correction:** for independent zero-mean Gaussian
  samples, normalized frame power follows a chi-squared distribution. Divide the
  lower-decile power by `chi2.ppf(0.1, 320)/320` before converting to dBFS. This
  analytic correction addresses finite-frame quantile bias; no parameter was fit
  to recordings. On four stationary white-noise fixtures, mean absolute error was
  **0.0155 dB**, versus **0.4559 dB** for the uncorrected statistic. Colored,
  correlated, non-Gaussian and speech-filled frames violate its assumptions.
  It is not an SNR estimator. Distribution implementation:
  [SciPy chi-squared documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html).
- **Speech activity:** the existing -50 dBFS energy threshold is evaluated only as
  a diagnostic surrogate. It marks all stationary-noise frames active in these
  fixtures, demonstrating why energy activity must not be labeled speech.
  An optional WebRTC VAD adapter uses 20 ms mono PCM16 frames at 16 kHz, mode 2,
  and records package provenance and input clipping. Its actual package is not
  installed locally, so the receipt reports unavailable rather than synthetic
  VAD scores. The adapter's mocked conversion test verifies mechanics only.
  API reference: [upstream Python wrapper](https://github.com/wiseman/py-webrtcvad).

The initial noise comparison exposed lower-tail bias. The Gaussian correction
was added from its distributional formula rather than tuning an offset to the
fixtures. Its improved fixture arithmetic does not justify a production change.

## Annotation-aware evaluation

The comparison runner accepts only development manifests:

```sh
.venv/bin/python scripts/benchmark_methods.py --manifest docs/method-benchmark.example.json
```

[The example](method-benchmark.example.json) uses a known historical recording and
deliberately has **null** speech, level-event and noise-reference annotations.
It exercises file loading and prior-use tracking, not accuracy measurement.
Null means unknown. An empty interval list means independently verified absence
of that event type; the runner never converts one into the other.

Each supplied case requires an ID, namespaced connected `source_group`, local
audio path, source byte SHA-256 and `annotation_provenance`. Optional
`speaker_group` and `source_references` retain further identities in prior-use
tracking. Group all speakers, cloning references and recording derivatives
conservatively before evaluation. Annotation fields are:

```json
{
  "speech_intervals": null,
  "level_events": null,
  "noise_reference_dbfs": null
}
```

Known intervals use `{"start": 1.0, "end": 2.0}` in source seconds. Noise
references require a justified ground-truth noise level; total recording RMS
must not be substituted for one. The common representation is arithmetic mono
at 16 kHz. Complete 20 ms frames are used for activity/noise statistics; any
incomplete tail is reported. Speech reference intervals are clipped to that
same analyzed coverage.

Event scoring maximizes the number of one-to-one matches at IoU >= 0.5, then
total valid IoU. Duplicate predictions cannot each claim the same reference.
The assignment uses [SciPy's linear assignment implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
Counts, precision, recall, F1, false events per fully negative hour, and boundary
errors for matched events are explicit. Missing-class quantities are null.

Paired micro-F1 differences use 10,000 connected-source bootstrap draws. Invalid
class-empty draws are counted, not filled with invented scores. The fixture
interval collapses to a fixed gain because the constructed groups share an
identical error pattern; it says nothing about population precision. Natural
annotation review and independent data remain required.

## Durable prior-use tracking

Supplied-file comparisons automatically write content-addressed receipts under
`data/evaluation/method-runs/`, including source byte and decoded-sample identities.
Repeated identical comparisons reuse identical receipts. No source audio is
saved or modified. Synthetic fixture runs do not create prior-use records.

Generate a fresh registry after conducting development comparisons:

```sh
.venv/bin/python scripts/inventory_corpora.py --registry-output /tmp/current-prior-use.json
```

The registry includes four historical manifests plus recorded local comparisons.
It checks comparison receipt content hashes. Qualification checks its historical
manifest/mapping hashes and complete comparison-receipt file set against the
current workspace. New comparisons make an older registry stale; regenerate to
a new filename before qualification. This registry does not cover undocumented
manual listening, unrecorded external experiments or upstream model training.

## Prospective corpus qualification

```sh
.venv/bin/python scripts/qualify_corpus.py docs/prospective-corpus.example.json \
  --prior-use /tmp/current-prior-use.json --output /tmp/qualification-example.json
```

This example is **expected to exit 1 / BLOCKED**. The
[qualification receipt](corpus-qualification-example-v1.json) records prior-use
overlap, pending audits, missing partitions/holdouts and insufficient sample
support. It is a rejection demonstration, not an acquired independent corpus.

For actual candidates, the tool checks:

- Required metadata, unique IDs, namespace-qualified identities, valid labels and
  generator families, file existence, byte hashes, duration/header limits.
- Transitive speaker/source/reference/parent/byte connections across partitions,
  parent cycles, conflicting byte-identical reference labels and prior-use overlap.
- Versioned, hash-verified review files for perceptual duplicates, label provenance,
  training exposure and previous use. Missing reviews and unknown/overlapping
  training exposure block structural qualification.
- Development, calibration and locked-test presence, declared held-out generator
  families and domains, and contract sample minimums per critical condition.
- Current prior-use registry evidence and comparison history.

Use canonical identity strings matching the history registry, such as
`librispeech:speaker:8230` and `librispeech:utterance:8230-279154-0026`.
`original_sha256` identifies the supplied case file's bytes; `parent_id` connects
derivatives to other cases. A derivative's own file hash must still match.

Successful structural checks verify metadata relationships and review-file
integrity. They do not independently prove a reviewer's assertions or establish
detection accuracy. Unknown provenance is excluded and blocks confirmatory use.
No new corpus has qualified, no locked labels have been scored, and no model or
anomaly method has been promoted.

## Checks and remaining work

Focused tests cover transitive leakage, historical reuse, stale registries,
tampered review/audio files, sample support, duplicate event matching, unavailable
VAD, explicit unknown annotations, decoded-duplicate grouping, durable receipts,
and PCM adapter mechanics. The studio CI workflow now includes these tests,
source arithmetic fixtures, comparison smoke checks and frontend export checks.
Hosted CI has not been run from this local workspace.

Next: acquire provenance-backed independent recordings and reviewed speech,
noise and event annotations, then run the development comparisons. Any candidate
promotion still requires the frozen contract and a separate locked evaluation.

## Subsequent annotated development study

Actual WebRTC and pinned Silero comparisons are now available in the
[VAD development study](vad-development-study.md). This supersedes the earlier
availability limitation for the isolated research runtime. Independent annotated
qualification and locked evaluation remain outstanding.
