# Alternative classifier: rejected

The installed Wav2Vec2-XLSR classifier was evaluated as a bounded local
alternative, without downloads, fitting, threshold search or product changes.
It is not the newer noise-trained WavLM candidate described below.

The same 45 publisher-labeled VoxENES originals and five deterministic conditions
were used. Source bytes and decoded inputs were verified. The pinned installed
checkpoint was checksum-verified, loaded offline, and evaluated at its existing
0.5 threshold with the current engineering quality/abstention policy. The
single-model candidate has no component-agreement safeguard. Product-specific
extra perturbation checks were not run for either core comparison.

| Condition | Current fusion raw AI detected | Alternative raw AI detected | Alternative human false alerts | Alternative wrong definitive results |
| --- | ---: | ---: | ---: | ---: |
| Original | 25/25 | 11/25 | 5/20 | 16/45 |
| Gain −12 dB | 25/25 | 11/25 | 5/20 | 16/45 |
| Added noise | 19/25 | 0/25 | 0/20 | 24/45 |
| Bandwidth 8 kHz | 23/25 | 9/25 | 5/20 | 18/45 |
| Vorbis round trip | 25/25 | 10/25 | 5/20 | 16/45 |

**Reject the alternative.** Its zero noisy-human false alerts are accompanied
by failure to detect every noisy synthetic clip. Across all 225 correlated rows,
it makes 90 wrong definitive assessments (12 false synthetic indications and
78 false human indications). It must not replace or corroborate the live model.
The complete [comparison receipt](core-alternative-classifier-v1.json) includes
per-record scores, source/model/code hashes, environment and a durable prior-use
receipt. Total inference plus model-load runtime was 90.88 seconds on the recorded
environment, with one Torch thread. Cold/warm latency and peak memory were not
separately measured; this is not a resource-promotion result.

These are repeated development observations on 45 originals, conservatively one
unresolved connected source group. They do not establish independent accuracy,
training independence, calibrated probabilities or general performance. The
fusion counts are the previously frozen core scores, not a fresh product run.

## Newer WavLM candidate preflight

The publisher of
[WavLM + AASIST](https://huggingface.co/eliya/forensics_0.3B_base_deepfake_classifier)
describes noise-augmented training. Pinned revision
`cfd11365b02167214dc3b392378809fef6ca3cfa` supplies a 1,269,932,956-byte safetensors
checkpoint and declares CC-BY-NC-4.0. No weights were downloaded or executed.
The [preflight receipt](core-classifier-preflight-v1.json) records inspected
source identities and the checkpoint SHA-256.

Its example code implicitly downloads a second backbone, allows unsafe pickle
fallback, and uses non-strict state loading. Any future isolated evaluation
must instantiate the architecture locally, use only hash-verified safetensors,
require strict state loading and reproduce the publisher preprocessing and score
direction. Large automatic acquisition is excluded by the existing evaluation
contract; commercial redistribution also cannot be assumed from this license.
Publisher benchmark claims have not been independently verified here.

## Reproduce the local comparison

Use installed models and acquired sources, and a new output path:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/screen_core_detector.py \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --output /tmp/new-alternative-screen.json --stress --detector wav2vec2
```

The research selector defaults to fusion. It changes no app settings or UI.
