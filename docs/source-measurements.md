# Source measurements 1.0

SignalTrace now measures decoded source channels at their original sample rate,
before channel mixing or resampling. The versioned results are returned as
`metrics.source_measurements` by the inspect API and under `diagnostics` in
forensic reports. New forensic reports use schema `forensics-1.2`.

## What is measured

For each original decoded channel: RMS amplitude/dBFS, minimum/maximum sample,
sample peak amplitude/dBFS, crest factor, DC offset, near-full-scale sample count
and fraction (`|sample| >= 0.999`), and over-full-scale sample count and fraction
(`|sample| > 1`). Amplitude 1 is the dBFS reference. Silent-channel dB values and
undefined ratios are represented by JSON null; linear amplitudes remain explicit.

Arithmetic mixdown energy is compared with mean original-channel power. The
energy ratio and attenuation expose channel interaction; complete cancellation
is represented explicitly, with null attenuation rather than an artificial
finite dB value. Ordinary uncorrelated channels also attenuate in an arithmetic
mean, so attenuation alone is not an anomaly or an authenticity judgment.

Calculations accumulate float64 statistics in bounded 65,536-frame chunks. The
input array is not modified. These are measurements of decoded samples, not
bit-exact recovery of encoded source precision or reconstructed true peaks.
Near-full-scale occupancy does not prove clipping.

## Relationship to the frozen detector

Existing top-level diagnostics and waveform/event analysis continue to use the
arithmetic mono signal at 16 kHz. They now carry `analysis_representation`
metadata documenting the distinction. Original-channel RMS/clipping and stereo
correlation retain their existing definitions. The studio summary labels its
level **Mono RMS level**, while the source-measurement table shows native channel
levels explicitly.

No neural weights, window policy, score aggregation, threshold, model input or
abstention rule changed. One additional warning reports source-channel samples
above full scale when mono mixing hides those peaks. Existing near-full-scale
warning wording is corrected to describe a near-limit observation. Warnings are
visible in the studio and both signal/model warnings are retained in JSON evidence
exports; they do not alter classifier scores.

No new acoustic event detector has been promoted. Adaptive change detection,
speech activity and noise estimation still require annotated natural audio and
the contract's false-alarm/precision/recall gates.

## Reproduce the arithmetic checks

```sh
.venv/bin/python scripts/validate_source_measurements.py
.venv/bin/python -m pytest tests/test_source_measurements.py tests/test_corpus_inventory.py tests/test_signaltrace.py -q
node --test tests/frontend-evidence.test.cjs
```

[Validation receipt](source-measurement-validation-v1.json): **18 analytic sine
fixtures** at 8, 16, 44.1, 48, 96 and 192 kHz, with maximum RMS/sample-peak/crest
error **0.00005488 dB**, below the frozen **0.1 dB** tolerance. All five receipt
checks pass, including source peaks hidden by opposite-phase mixing, over-range
sample counts and explicit silent-channel levels.

Additional regression tests cover multichannel chunked arithmetic, DC offset,
native high-frequency energy lost from the derived representation, cancellation,
invalid inputs and inspect-API serialization. Frontend tests exercise the shipped
warning renderer and evidence-export handler, including escaped warning text and
combined warning retention. A browser check verified the rendered native table,
method labels and persistent warning display.

These checks establish arithmetic behavior on controlled signals. They do not
establish speech detection, edit localization or real-world anomaly accuracy.
