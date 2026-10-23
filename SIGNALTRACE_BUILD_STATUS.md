# Expanded external audio acquisition

Expanded to 900 unscored clips from 600 declared speakers. Independently
verified 300 genuine calibration speakers and 300 genuine evaluation speakers,
100/200 synthetic speakers respectively, and no speaker leakage between them.
All 40 pilot reservations are preserved. No model or threshold changes.
See [expanded report](docs/asvspoof5-expanded.md) for integrity evidence and
unresolved training/identity exposure limits.

# External audio acquisition

Acquired and integrity-audited40 unscored ASVspoof5 recordings from40 declared
speakers under an80MiB download cap. Balanced reserved partitions remain
unscored; training exposure and cross-corpus identity qualification are pending.
See [pilot report](docs/asvspoof5-pilot.md). No app/model changes.

# Capability investigation update

A complementary WavLM detector recovered the remaining noisy v4 miss without
additional raw genuine false alerts under a frozen OR comparison. This is spent
regression evidence, not production promotion. Short AI inserts remain a gap;
negative UI results now say No clear AI signal. See
[capability report](docs/detector-capability-update.md).

# Compact noise-augmented classifier decision

RawNetLite was tested and rejected: 6/25 noisy synthetic detections and 4/20
noisy human false alerts. Its small size and speed do not justify its errors.
No app/model/design changes. See [decision](docs/core-rawnetlite-decision.md).

# Alternative classifier decision

The installed third classifier failed the paired modern-voice screen: 11/25
clean and 0/25 noisy synthetic detections. It is rejected. A newer WavLM
candidate received source/license/size preflight only; no large weights were
downloaded. See [classifier comparison](docs/core-classifier-comparison.md).
The app keeps the current fusion detector and simple workflow.

# Noise-robustness candidate decision

Three development preprocessing candidates were tested and rejected. Spectral
reduction did not improve 19/25 noisy synthetic detections; low-pass fell to 15/25.
MetricGAN reached 25/25 but falsely flagged 20/20 noisy human clips, including
14 wrong definitive indications. It is not enabled in the app. Model weights,
thresholds and the human-corroboration safeguard remain unchanged.
See [noise improvement study](docs/core-noise-improvement.md).

# Core detector update

Direct modern-generator screening now covers 45 publisher-labeled originals
across five newer English TTS systems and five conditions (225 correlated rows).
Clean screening flagged 25/25 synthetic clips with 0/20 human raw false alerts;
added noise exposed two wrong human indications, confirmed through the product.

Policy 1.1 requires component corroboration before indicating human speech.
Both known errors now become inconclusive; development replay retains all 104
synthetic indications while definitive coverage decreases from 82.7% to 76.9%.
A broader disagreement rule and a worse bounded-logit fusion candidate were
rejected. Raw noise sensitivity still needs improvement. See
[core detector progress](docs/core-detector-progress.md) for full counts and limits.
The simple upload/assessment workflow is unchanged.

# Current product focus

SignalTrace is an AI-voice assessment tool: upload → automatic local detection →
clear result/reasons. Playback, raw scores and acoustic evidence are collapsed;
archived benchmarks are secondary. Random samples no longer load on startup.

The current refocus adds quality preflight that skips expensive model loading
for already-inconclusive recordings, makes failed perturbation stability checks
inconclusive, and prevents stale asynchronous responses from replacing a newer
recording's result. Frozen detector weights and score thresholds remain unchanged.
No detection-accuracy improvement or world-class claim is inferred from UI work.

## Earlier research implementation history

# SignalTrace

Implementation branch for the waveform-first Audio Evidence Studio. Existing detector and evaluation code remains preserved. Work in progress.

Methods improvement milestone 1: baseline integrity/archived replay audit and
prospective evaluation contract implemented. See
[baseline methods audit](docs/baseline-methods-audit.md) and
[evaluation contract v1](docs/evaluation-contract-v1.json).
Broader detector validation, new locked data, calibrated reliability and
manipulation localization remain research work; no world-class claim is established.

Milestone 2: local connected-identity corpus inventory and native-rate,
per-channel source measurements implemented and validated on analytic fixtures.
The studio displays measurement provenance and persistent quality warnings;
JSON exports retain source measurements and signal/model warnings. See
[source measurements](docs/source-measurements.md) and
[corpus acquisition roadmap](docs/corpus-acquisition-roadmap.md).
No new anomaly rules or neural detector were promoted.

Milestone 3: prospective corpus qualification, development-only method comparison,
and durable prior-use receipts implemented. Experimental adaptive level changes,
noise statistics and an optional WebRTC adapter remain outside production findings.
See [method research](docs/method-research.md) for results, assumptions and limits.
Independent annotated data and actual VAD performance remain unverified.

Milestone 4: bounded acquisition and actual annotated VAD comparison completed.
Silero raw-frame F1 is 93.81% versus WebRTC 89.37% on 30 development clips;
unresolved identities conservatively form one group. ASVspoof 5 protocols were
audited without acquiring its audio. See [VAD development study](docs/vad-development-study.md).
No independent-test qualification or production promotion is claimed.

Milestone 5: source-verified development speech-review packet preparation added.
All 30 TEN clips have a local packet with neutral IDs and pending annotations;
publisher labels and model predictions are withheld from the reviewer directory.
No independent review has occurred. See the study's review-preparation section.

Milestone 6: completed-review validation and paired frame agreement implemented,
with explicit pending-label rejection, undefined degenerate kappa and an
adjudication queue that also captures subframe interval differences and notes.
Only constructed review fixtures have been compared; actual review remains pending.

Milestone 7: explicit speech adjudication and development-manifest export added.
Disputes require final intervals, rationale and a distinct adjudicator identifier;
exports bind original review/mapping/source hashes and retain development status.
Constructed fixtures verify export-to-benchmark compatibility. Real reviews and
adjudication remain pending; no model promotion or new accuracy result is claimed.

Milestone 8: reusable verified Silero research session and local CPU resource
benchmark implemented. Speech outputs/raw scores are identical on all 30 clips.
Fresh-process resource runs separate cold setup, warmup, decode and warm latency;
see [runtime study](docs/vad-runtime-study.md). Full CPU-model metadata is unavailable
under the sandbox. Resource observations do not qualify the corpus or promote VAD.

Milestone 9: deterministic automated VAD stress suite generated and scored:
180 rows from 30 originals across six conditions, preserving one conservative
source group. Quiet-input and constructed-noise failures of energy activity are
measured; fixed Silero retains more inherited speech frames than WebRTC here.
See [automated stress study](docs/vad-stress-study.md). No intervention, new audio
acquisition, independent-label claim or production promotion was required.

Milestone 10: automated channel/codec/short-context study scored 240 development
rows from the same 30 originals. Raw Silero recall fell to 69.46% on 0.5-second
center crops; methods are not uniformly interchangeable. Existing diagnostic
warnings passed 120/120 cancellation and short-context checks. See
[channel stress study](docs/vad-channel-stress-study.md). No production changes
or independent-validation claims were introduced.
