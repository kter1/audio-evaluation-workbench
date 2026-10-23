# Local VAD resource study

The research runner now loads one verified Silero model per comparison and resets
its recurrent state before every recording. All speech outputs, including raw
scores, match the preceding per-recording-load run exactly on all 30 TEN clips.
The [new comparison receipt](vad-development-v3.json) preserves this run without
rewriting earlier evidence. Production scoring remains unchanged.

Fresh sequential processes measured each method on the same 30 development clips,
with three timed passes (90 calls), one excluded first-clip warmup, CPU execution
and OMP/MKL/OpenBLAS thread settings of 1. Silero also explicitly sets Torch intra-
and inter-op threads to 1. These repeats do not add independent recordings.

| Method | Warm total/audio time | Warm latency p50 / p95 | Peak process RSS |
| --- | ---: | ---: | ---: |
| Energy diagnostic | 0.0000115 | 0.107 / 0.154 ms | 0.118 GiB |
| WebRTC mode 2 | 0.000164 | 1.536 / 2.162 ms | 0.118 GiB |
| Silero raw | 0.003504 | 32.642 / 50.415 ms | 0.286 GiB |

Silero cold adapter setup took 0.936 s, including Torch import/configuration and
0.051 s for verified model loading. Its excluded first-recording warmup took
0.116 s. Cold loading is not included in the warm ratios. Warm calls include input
validation, per-recording state reset and output conversion. Decode/hash verification
is timed separately. The reusable session is sequential and not thread safe.

Receipts: [energy](vad-runtime-energy-v1.json),
[WebRTC](vad-runtime-webrtc-v1.json), [Silero](vad-runtime-silero-v1.json).
Each records individual timings, duration bins, environment, thread settings,
source hashes, input comparison hash and measured resource checks. Optional VAD
intervals are checked against the prior comparison on every timed call. The
original development manifest and decoded-input hashes must match that comparison.

Peak RSS includes Python, libraries and all decoded clips, not just the model.
The workers ran under the managed local sandbox on macOS arm64. CPU-model sysctl
queries were denied; the missing hardware fields are recorded explicitly. No
controlled-load, cross-machine, sustained-concurrency or long-recording study was
performed. First-recording warmup is not a claim that every possible JIT path is
warm. Timings are descriptive and may vary with scheduling and caches. The frozen
resource limits were met in these local runs only; corpus, accuracy, independence
and replication gates remain unmet.

## Reproduce

Use a new output path for each invocation and run methods sequentially in fresh
processes. No download occurs. A prior supplied-development comparison matching
the manifest is required; these already-consumed inputs remain development data.
For WebRTC, use the isolated runtime prepared in the annotation study.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=data/research-tools/webrtcvad-2.0.14.post1/site-packages \
.venv/bin/python scripts/benchmark_vad_runtime.py \
  --manifest data/evaluation/ten-vad-development-v1.json \
  --comparison docs/vad-development-v3.json --method silero \
  --silero-model data/research-tools/silero/066409f3d9db45a30d21dfcbf9147826ffdd7052/silero_vad.jit \
  --output /tmp/new-silero-runtime.json
```

For the other methods, change `--method` to `webrtc` or `energy`, omit
`--silero-model`, and choose a new output filename. Never interpret the energy
diagnostic's speed as validation of speech detection. See the
[annotated development study](vad-development-study.md) for accuracy limitations.
