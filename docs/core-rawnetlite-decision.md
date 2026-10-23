# Compact noise-augmented classifier: rejected

The author-released augmented triple-domain focal-loss RawNetLite checkpoint was
selected before scoring as a small, waveform-native, noise-augmented alternative.
The [official repository](https://github.com/adipiz99/RawNetLite) declares MIT;
its [paper](https://arxiv.org/abs/2504.20923) describes training with FakeOrReal,
AVSpoof2021 and CodecFake, including waveform augmentation. Publisher performance
claims did not determine acceptance. Search also found larger or noncommercial
alternatives; none of those checkpoints were acquired in this round.

The single checkpoint is 987,460 bytes. Total acquisition, including architecture,
preprocessing, label mapping, tester and license, was 1,006,560 bytes under the
1.1 MB cap. Every file was verified against a pinned Git blob identity.
[Acquisition specification](rawnetlite-acquisition-v1.json).

The architecture is vendored unmodified with the author's license. The adapter
verifies model/source identities before import or deserialization, loads only
restricted `weights_only=True` tensors, and requires a strict state match.
No packages, training jobs, paid services or app options were added.

## Fixed paired development result

Same preselected 45 publisher-labeled English recordings, same five constructed
conditions, one unresolved source/speaker group. No threshold tuning or family
checkpoint search. Threshold 0.5, native label 1=fake, label 0=real. Quality and
engineering abstention checks are applied; extra product perturbation checks are
not run. This single-model candidate has no component-consensus safeguard.

| Condition | Raw AI detected | Human false alerts | Wrong definitive results | Inconclusive |
| --- | ---: | ---: | ---: | ---: |
| original | 6/25 | 5/20 | 15/45 | 16/45 |
| gain_minus12db | 6/25 | 5/20 | 15/45 | 16/45 |
| added_noise20db | 6/25 | 4/20 | 13/45 | 20/45 |
| bandwidth8k | 5/25 | 2/20 | 15/45 | 15/45 |
| vorbis_roundtrip | 6/25 | 5/20 | 16/45 | 15/45 |

**Reject for production and corroboration.** The existing fusion flags 25/25
clean and 19/25 noisy synthetic clips, versus 6/25 and 6/25 here. Current raw human
false alerts are 0/20 clean and 1/20 noisy, versus 5/20 and 4/20 here.
The candidate makes 74 wrong definitive assessments across 225 correlated rows:
5 wrong synthetic indications on human clips and 69 wrong human indications on
synthetic clips. Speed does not compensate for these errors. App weights,
thresholds, design and upload flow remain unchanged.

[Full results](core-rawnetlite-screen-v1.json) record scores, source hashes,
model-input hashes, model/source identities and a content-addressed prior-use
receipt. Historical reports remain unchanged.

## Preprocessing and resource checks

The shared runner first reconstructs the existing center-at-most-4-second
mono16k screen crop. The author path then peak-normalizes the entire supplied
crop, takes its first 48,000 samples, and right-zero-pads shorter inputs. This
3-second context differs from the fusion input and is explicitly recorded.
The shared SciPy mono16 conversion is retained; no equivalence to the author's
TorchAudio resampler at other rates is claimed. Author preprocessing at 16 kHz
matches exactly on three fixtures of 8,000, 48,000 and 64,000 samples, including
a peak in the trimmed tail. Exact digital silence is intentionally defined as
zero instead of the upstream zero/zero. No evaluated crop was digitally silent.

The candidate has 240,769 parameters. On the recorded macOS CPU environment,
with one Torch thread, 225 calls took 8.40 seconds total: p50 36.45 ms, p95 41.74 ms.
Score-only real-time factor was 0.00986 against supplied crop duration, or 0.01245
against fixed 3-second model input duration. Entire process peak RSS was 0.431 GiB.
Verified model construction/load took 0.013 seconds; the full screen took 10.39
seconds. The first call is included in these latency quantiles; this is not a
cold-filesystem-cache or warmed product-serving benchmark. Shared decoding,
transforms and diagnostics are excluded from score timing but included in full
screen wall time. Resource feasibility passes the internal numerical bounds,
while accuracy fails badly; this is not model promotion.

These are development-only observations, not independent validation, population
risk estimates or a world-class claim. Source/speaker/training overlap is unknown.
Repeated conditions do not increase independent sample support.

## Reproduce

With acquired screen inputs and an installed Torch runtime, use a new output:

```sh
.venv/bin/python scripts/setup_research_rawnetlite.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/screen_core_detector.py \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --output /tmp/new-rawnetlite-screen.json --stress --detector rawnetlite
.venv/bin/python -m pytest tests/test_research_rawnetlite.py tests/test_core_screen.py -q
```

The lightweight candidate restriction has now excluded an actual compact,
noise-augmented author checkpoint on accuracy. Further progress requires a
more capable detector and better independent data, rather than additional
preprocessing or weak-model votes. Keep the current simple product intact.
