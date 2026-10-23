# Automated channel, codec and short-context study

Eight fixed conditions were run on the same 30 existing development clips:
240 case rows, still only one conservative connected source group. No new audio,
reviewer, model or threshold was introduced. All 30 original-condition speech
outputs exactly match the preceding full-clip comparison.

Each cell gives speech recall / nonspeech false-positive rate, in percent:

| Condition | Energy diagnostic | WebRTC mode 2 | Silero raw |
| --- | ---: | ---: | ---: |
| Original | 99.06 / 88.95 | 94.10 / 50.22 | 93.79 / 18.77 |
| 8 kHz bandwidth round trip | 98.75 / 85.56 | 94.04 / 50.03 | 92.74 / 17.93 |
| Stereo, one empty channel | 97.54 / 79.72 | 88.14 / 32.25 | 93.00 / 19.17 |
| Vorbis round trip | 99.10 / 89.07 | 94.03 / 49.91 | 94.09 / 18.77 |
| Center crop, 0.5 s | 100.00 / 92.20 | 91.63 / 31.91 | 69.46 / 12.06 |
| Center crop, 1 s | 99.83 / 90.13 | 90.64 / 40.46 | 81.10 / 14.14 |
| Center crop, 2 s | 99.53 / 87.59 | 89.49 / 43.80 | 85.84 / 21.97 |

The eighth condition is opposite-phase stereo. Every method receives exact zero
mono samples after the existing arithmetic mixdown, although the original source
channels retain audio. Its raw recall/FPR are both zero. This is a constructed
transport failure, not a fair ranking of VAD accuracy. The
[full receipt](vad-channel-stress-development-v1.json) retains those counts
explicitly, together with all per-condition frame counts and source hashes.

Silero remains similar under the tested bandwidth/codec changes, but its raw
0.5-second speech recall is substantially lower than full-clip recall and lower
than WebRTC on these fixed crops. No method is uniformly superior across the
reported conditions. No automatic threshold change, padding policy or production
promotion follows from inspecting these development results.

## Construction and timing

- **Bandwidth:** downsample 16→8 kHz with SciPy `resample_poly`, then upsample
  8→16 kHz and retain the original frame count. This is a constructed bandwidth
  restriction, not a measured telephone or device channel. Filter edge effects
  remain possible; inherited speech timing is not newly reviewed.
- **One empty channel:** pair the source with zeros. Arithmetic mono mixing
  yields exactly half amplitude (−6.02 dB).
- **Opposite-phase stereo:** pair the source with its negative. Original channels
  retain their energy while arithmetic mono mixing cancels exactly.
- **Codec:** encode/decode OGG/Vorbis using SoundFile 0.13.1 and libsndfile 1.2.2,
  compression level 0.5. All 30 decoded outputs retain original sample count,
  rate and zero global waveform-correlation lag. Nonzero lag blocks generation;
  there is no silent label realignment. Encoded assets, encoded hashes and decoded
  float32 WAV assets are retained locally. OGG container bytes may vary across
  encodes; recorded assets/hashes identify this run. Global correlation does not
  prove identical local perceptual boundaries.
- **Short context:** crop the center by sample count, independent of labels and
  model predictions, to 0.5, 1 and 2 seconds. Intersect source speech intervals
  with each crop and subtract its start offset. Unknown labels remain unknown;
  empty intersections remain explicit nonspeech references. Shorter originals
  would use all available samples rather than inventing padding.

All cases preserve source/speaker grouping and parent-byte references. Output is
bounded by the original preparation limits and 256 MiB storage cap; this run
created about 109.4 MB. Raw and encoded assets remain in ignored local directories.
Original labels are inherited latent timing, not new independent truth about
transformed audibility. Repeated derivatives do not increase independent sample
support. No population confidence interval or authenticity/edit-origin claim is
made.

## Existing diagnostic checks

The [transport audit](vad-transport-audit-v1.json) passed **120/120 cases**:
30 exact-cancellation fixtures retain nonzero source-channel energy and trigger
both opposite-phase and insufficient-active-signal warnings; all 90 short crops
trigger the existing minimum-context warning. The audit verifies audio hashes,
source/condition coverage and actual diagnostic output. It does not run origin
models or establish end-to-end abstention behavior.

Tests also verify the audit fails its checks if warnings disappear, reject
misaligned codec output rather than repairing it, check high-frequency attenuation,
and check exact crop/channel arithmetic. No production diagnostic was modified.

## Reproduce

No network access or FFmpeg is required. A local SoundFile build supporting Vorbis
encoding and the compression-level option is required for this extended suite.
Use new output paths; existing inputs and evidence are preserved.

```sh
.venv/bin/python scripts/prepare_vad_stress.py --extended \
  --manifest data/evaluation/ten-vad-development-v1.json \
  --output data/research-corpora/vad-channel-stress-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=data/research-tools/webrtcvad-2.0.14.post1/site-packages \
.venv/bin/python scripts/benchmark_methods.py \
  --manifest data/research-corpora/vad-channel-stress-new/manifest.json \
  --silero-model data/research-tools/silero/066409f3d9db45a30d21dfcbf9147826ffdd7052/silero_vad.jit \
  --output /tmp/new-channel-stress.json
.venv/bin/python scripts/audit_vad_transport.py \
  --manifest data/research-corpora/vad-channel-stress-new/manifest.json \
  --output /tmp/new-transport-audit.json
```

The preceding [gain/noise/timing study](vad-stress-study.md) remains separate.
These automated checks can continue without manual review; independent label
correctness, training exposure and broader recording-domain coverage remain
unverified.
