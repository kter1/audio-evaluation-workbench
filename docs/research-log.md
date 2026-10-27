# Research log

Newest entries first. Each entry summarizes one piece of work and links to the
report that has the full counts, method, and limitations.

## Expanded external audio set

Expanded to 900 unscored clips from 600 declared speakers. Independently
verified 300 genuine calibration speakers and 300 genuine evaluation speakers,
100/200 synthetic speakers respectively, and no speaker leakage between them.
All 40 pilot reservations are preserved. No model or threshold changes.
See the [expanded report](asvspoof5-expanded.md) for integrity evidence and
unresolved training and identity exposure limits.

## External audio pilot

Acquired and integrity-audited 40 unscored ASVspoof 5 recordings from 40 declared
speakers under an 80 MiB download cap. Balanced reserved partitions remain
unscored; training exposure and cross-corpus identity qualification are pending.
See the [pilot report](asvspoof5-pilot.md). No app or model changes.

## Complementary detector investigation

A complementary WavLM detector recovered the remaining noisy v4 miss without
additional raw genuine false alerts under a frozen OR comparison. This is
regression evidence on previously used data, not a production change. Short AI
inserts remain a gap; negative results in the app now read **No clear AI signal**.
See the [capability report](detector-capability-update.md).

## Compact noise-augmented classifier: rejected

RawNetLite was tested and rejected: 6/25 noisy synthetic detections and 4/20
noisy human false alerts. Its small size and speed do not justify its errors.
No app or model changes. See the [decision](core-rawnetlite-decision.md).

## Alternative classifier: rejected

The installed third classifier failed the paired modern-voice screen: 11/25
clean and 0/25 noisy synthetic detections. A newer WavLM candidate received a
source, license, and size preflight only; no large weights were downloaded.
See the [classifier comparison](core-classifier-comparison.md). The app keeps
the current fusion detector.

## Noise-robustness preprocessing: rejected

Three development preprocessing candidates were tested and rejected. Spectral
reduction did not improve 19/25 noisy synthetic detections; low-pass filtering
fell to 15/25. MetricGAN reached 25/25 but falsely flagged 20/20 noisy human
clips, including 14 wrong definitive indications. It is not enabled in the app.
Model weights, thresholds, and the human-corroboration safeguard are unchanged.
See the [noise improvement study](core-noise-improvement.md).

## Core detector screening and policy 1.1

Direct modern-generator screening covers 45 publisher-labeled originals across
five newer English TTS systems and five conditions (225 correlated rows). Clean
screening flagged 25/25 synthetic clips with 0/20 human raw false alerts; added
noise exposed two wrong human indications, confirmed end to end in the app.

Policy 1.1 requires both component models to agree before indicating human
speech. Both known errors now become Inconclusive; development replay retains
all 104 synthetic indications while definitive coverage decreases from 82.7% to
76.9%. A broader disagreement rule and a bounded-logit fusion candidate were
rejected. Raw noise sensitivity still needs improvement. See
[core detector progress](core-detector-progress.md).

## Product focus

SignalTrace is organized around one workflow: upload, automatic local
detection, and a clear result with reasons. Playback, raw scores, and acoustic
evidence are collapsed by default; archived benchmarks are secondary.

A quality preflight skips model loading for recordings that are already
Inconclusive, failed perturbation-stability checks make a result Inconclusive,
and stale asynchronous responses can no longer replace a newer recording's
result. Detector weights and score thresholds are unchanged, and no accuracy
improvement is inferred from interface work.

## Methods milestones

1. **Baseline audit.** Baseline integrity and archived replay audit, plus a
   prospective evaluation contract. See the
   [baseline methods audit](baseline-methods-audit.md) and
   [evaluation contract v1](evaluation-contract-v1.json). Broader validation, new
   locked data, calibrated reliability, and manipulation localization remain open.
2. **Source measurements.** Local corpus inventory and native-rate, per-channel
   source measurements, validated on analytic fixtures. The app shows
   measurement provenance and persistent quality warnings; JSON exports keep
   source measurements and warnings. See
   [source measurements](source-measurements.md) and the
   [corpus acquisition roadmap](corpus-acquisition-roadmap.md).
3. **Corpus qualification.** Prospective corpus qualification, development-only
   method comparison, and durable prior-use receipts. Experimental adaptive level
   changes, noise statistics, and an optional WebRTC adapter stay outside
   production findings. See [method research](method-research.md).
4. **Annotated VAD comparison.** Silero raw-frame F1 is 93.81% versus WebRTC
   89.37% on 30 development clips; unresolved identities conservatively form one
   group. ASVspoof 5 protocols were audited without acquiring its audio. See the
   [VAD development study](vad-development-study.md).
5. **Speech-review packets.** All 30 development clips have a local review packet
   with neutral IDs and pending annotations; publisher labels and model
   predictions are withheld from the reviewer directory.
6. **Review agreement.** Completed-review validation and paired frame agreement,
   with explicit pending-label rejection, undefined degenerate kappa, and an
   adjudication queue that also captures sub-frame interval differences and notes.
7. **Adjudication and export.** Disputes require final intervals, a rationale,
   and a distinct adjudicator ID; exports bind the original review, mapping, and
   source hashes. Real reviews and adjudication remain pending.
8. **Runtime benchmark.** A reusable verified Silero session and local CPU
   resource benchmark. Speech outputs and raw scores are identical on all 30
   clips. See the [runtime study](vad-runtime-study.md).
9. **Stress suite.** 180 rows from 30 originals across six conditions. Quiet-input
   and constructed-noise failures of energy-based activity are measured; Silero
   retains more inherited speech frames than WebRTC here. See the
   [automated stress study](vad-stress-study.md).
10. **Channel and short-context stress.** 240 development rows from the same 30
    originals. Raw Silero recall fell to 69.46% on 0.5-second center crops; the
    methods are not interchangeable. Existing diagnostic warnings passed 120/120
    cancellation and short-context checks. See the
    [channel stress study](vad-channel-stress-study.md).
