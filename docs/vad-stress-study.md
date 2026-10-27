# Automated development robustness study

A fixed deterministic suite tested 30 existing TEN clips under six conditions:
original, −12 dB gain, −24 dB gain, constructed noise at two levels, and an exact
137 ms silence prefix. There are 180 case rows but still only 30 original clips
and one conservative connected source group. No reviewer, download or threshold
tuning was required. These derivatives are not new independent recordings.

Each cell gives speech recall / nonspeech false-positive rate, in percent:

| Condition | Energy diagnostic | WebRTC mode 2 | Silero raw |
| --- | ---: | ---: | ---: |
| Original | 99.06 / 88.95 | 94.10 / 50.22 | 93.79 / 18.77 |
| Gain −12 dB | 91.42 / 53.70 | 84.01 / 28.86 | 91.18 / 17.93 |
| Gain −24 dB | 38.49 / 4.94 | 69.51 / 17.81 | 81.84 / 13.83 |
| Mixture/noise 20 dB | 100.00 / 100.00 | 91.52 / 43.64 | 93.07 / 17.96 |
| Mixture/noise 10 dB | 100.00 / 100.00 | 90.14 / 54.14 | 92.55 / 18.61 |
| Silence prefix 137 ms | 99.07 / 84.34 | 95.16 / 54.08 | 93.80 / 18.95 |

These results expose failure of energy activity as a speech proxy under noise
and quiet input. Silero retains more inherited speech frames under attenuation,
with lower nonspeech activation under constructed noise than WebRTC here. No
model or threshold has been promoted. All 30 original-condition speech outputs
match the preceding comparison exactly. Full counts, frame F1 and source hashes
are in the [stress receipt](vad-stress-development-v1.json).

## Construction and interpretation

Inputs must be development-only, source-hash-verified mono 16 kHz WAV, 0.5–60 s
each, with at most 50 originals and 600 total source seconds. Preparation publishes
a new directory only after successful completion; output is capped at 256 MiB.
This run used about 84.5 MB including manifest. Raw derivatives remain in ignored
local data directories; no source audio is redistributed in the documentation.

Gain changes preserve sample timing. Noise uses per-source/condition SHA-derived
seeds and explicit NumPy PCG64, normalized to a specified ratio of original
whole-mixture RMS to added-noise RMS. These are not clean-speech SNR measurements:
the original recordings may already contain noise. Float32 WAV output avoids
normalization and storage clipping. The WebRTC PCM adapter still clips values
outside its PCM16 range: 2,720 samples in the 20 dB noise condition and 4,140 in
the 10 dB condition exceed that range, as recorded in transformation metadata and
adapter results. Other conditions contain no such samples in this run.

The prefix inserts exactly 2,192 zero samples and shifts original intervals by
2,192/16,000 seconds. Endpoints are bounded to decoded duration to avoid floating
point overshoot. This deliberately changes alignment relative to 20 ms scoring
frames and 32 ms model chunks. Prefix silence is constructed; original labels are
inherited, not independently confirmed. Unknown annotations remain unknown.

Original speaker/source identities and parent-byte references remain connected
across all derivatives. Metrics are separated by condition; pooled derivative
counts must not be interpreted as independent sample support. No population
confidence interval is claimed. Quiet speech or added noise may alter perceived
audibility, so inherited latent speech timing is not newly reviewed perceptual
truth. The study establishes behavior under specified transformations, not
real-world domain coverage, authenticity detection or edit localization accuracy.

## Reproduce without intervention

Use a new output directory/file for each run; existing outputs are protected.
The isolated WebRTC runtime and pinned Silero model must already be present.

```sh
.venv/bin/python scripts/prepare_vad_stress.py \
  --manifest data/evaluation/ten-vad-development-v1.json \
  --output data/research-corpora/vad-stress-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=data/research-tools/webrtcvad-2.0.14.post1/site-packages \
.venv/bin/python scripts/benchmark_methods.py \
  --manifest data/research-corpora/vad-stress-new/manifest.json \
  --silero-model data/research-tools/silero/066409f3d9db45a30d21dfcbf9147826ffdd7052/silero_vad.jit \
  --output /tmp/new-vad-stress-result.json
```

The benchmark automatically includes per-condition summaries and saves a durable
prior-use receipt. Missing conditions, duplicate source/condition pairs and
mismatched case coverage block summarization. Tests cover deterministic noise
scaling, exact prefix sample counts, unknown/empty annotations, source identity,
atomic failure cleanup, interval-end rounding and grouping through the runner.

Automated robustness work can continue without label adjudication. Independent
label correctness, physical speaker identities and upstream training exposure
remain unverified; automation is not represented as evidence resolving them.

## Subsequent channel and short-context extension

A separate [channel/codec/short-context study](vad-channel-stress-study.md) tests
bandwidth restriction, stereo mixing, Vorbis and fixed center crops. It records
short-clip limitations and verifies existing transport/context diagnostic warnings.
