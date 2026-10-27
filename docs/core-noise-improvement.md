# Noise robustness: controlled preprocessing experiments

The live AI-voice detector remains simple and keeps its existing frozen model
weights, thresholds and human-corroboration safeguard. These experiments test
whether preprocessing can improve its noisy-synthesis sensitivity. No candidate
is enabled automatically by opening or uploading to the app.

## Cohort and fixed comparison

The same preselected 45 publisher-labeled English development recordings are
used: 20 human and 5 each from Chroma, GLM-TTS, Qwen3-TTS, VibeVoice and VoxCPM.
Each has its unchanged center crop and the previously specified 20 dB
whole-mixture/constructed-white-noise condition. This ratio is not clean-speech
SNR. The resulting 90 input conditions remain paired derivatives of 45 originals;
unresolved identities conservatively remain one source group.

Both classes receive identical processing. Each candidate uses the frozen fusion
threshold and current engineering decision policy. There is no threshold search,
classifier retraining, generator-specific switch or selection by label. Baseline
raw scores are reused after byte/decoded-input reconstruction checks; candidate
scores come from fresh neural inference. Quality and corroboration decisions are
recomputed consistently. The product's extra perturbation-stability check is not
run here. Therefore these are core development comparisons, not full product
accuracy estimates or independent validation.

## Simple candidates: rejected for sensitivity improvement

Two fixed candidates were defined before scoring:

- **Spectral floor:** 512-sample Hann STFT, hop 128, per-frequency temporal power
  quantile 0.2 as the noise estimate, power subtraction, magnitude-gain floor 0.2,
  inverse STFT and exact output length. Continuous speech may be removed as noise.
- **Low-pass control:** order-6 Butterworth at 3.4 kHz with zero-phase filtering,
  matching the type of probe already used in the product's stability checks.

| Condition/method | Raw AI detected | Raw human false alerts | Definitive AI indications | Inconclusive |
| --- | ---: | ---: | ---: | ---: |
| Clean, unchanged | 25/25 | 0/20 | 23 | 6/45 |
| Clean, spectral floor | 25/25 | 0/20 | 23 | 6/45 |
| Clean, low-pass | 22/25 | 0/20 | 21 | 14/45 |
| Added noise, unchanged | 19/25 | 1/20 | 15 | 21/45 |
| Added noise, spectral floor | 19/25 | 0/20 | 14 | 16/45 |
| Added noise, low-pass | 15/25 | 0/20 | 11 | 25/45 |

Neither improves noisy AI sensitivity. The spectral candidate improves human
specificity/coverage on these examples but loses one clear AI indication; the
low-pass candidate damages both clean and noisy detection. Both are rejected as
solutions to the stated sensitivity problem. No wrong definitive assessments
occurred under these candidate conditions, with policy 1.1 applied. Full paired
rows, counts and hashes are in the [simple-candidate receipt](core-noise-candidates-v1.json).

## Third candidate: pinned neural speech enhancement

MetricGAN+ is evaluated separately as a third candidate, not added after looking
at individual neural-candidate scores. The pretrained
[SpeechBrain model](https://huggingface.co/speechbrain/metricgan-plus-voicebank)
is trained for English speech enhancement, not AI detection; its publisher does
not warrant performance on other datasets. The checkpoint is 7,586,021 bytes,
model revision `a196ce26b3bdace6fa1d819017584bdbcce462a8`, publisher-declared
Apache-2.0. Total model/source acquisition was 7,704,577 bytes. No SpeechBrain,
TorchAudio or other package was installed in the main environment.

The research-only Torch port follows the pinned upstream two-layer bidirectional
LSTM, learned sigmoid mask on log magnitudes, noisy phase, 512-point Hamming
STFT with constant boundary padding and hop 256, inverse transform and upstream
peak normalization. It verifies checkpoint Git blob identity before restricted
`weights_only=True` loading. The port handles full single recordings only; its
LSTM state is fresh per call. Exact digital silence is returned unchanged to
avoid epsilon/peak-normalization amplification.

[Port validation](metricgan-port-validation-v1.json) passed six noise/tone tensor
fixtures across 8,000, 16,001 and 64,000 samples with maximum absolute error **0.0**
against inspected upstream definitions at pinned SpeechBrain commit
`c0241278f93ef02cd5191f03caf4e5f8e1292bf6`. The reference uses a plain Torch LSTM
wrapper for full-length sequences. This is tensor-component parity, not a claim
of full installed-SpeechBrain package equivalence or speech quality. The
intentional exact-zero guard differs from upstream epsilon normalization.

The encoder can remove forensic artifacts or introduce new ones. Peak
normalization also changes amplitude diagnostics, so successful development
counts alone would not justify replacing production input handling.

## Reproduce

Inputs and the two existing AI detectors must already be installed. Choose new
output paths. No paid inference or new model training is used.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/evaluate_noise_candidates.py \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --baseline docs/core-modern-stress-v1.json \
  --output /tmp/new-simple-noise-candidates.json
.venv/bin/python scripts/setup_research_metricgan.py
.venv/bin/python scripts/validate_metricgan_port.py \
  --directory data/research-tools/metricgan/a196ce26b3bdace6fa1d819017584bdbcce462a8 \
  --output /tmp/new-metricgan-port-check.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/evaluate_noise_candidates.py \
  --acquisition data/research-corpora/voxenes-2026/acquisition.json \
  --baseline docs/core-modern-stress-v1.json \
  --metricgan-directory data/research-tools/metricgan/a196ce26b3bdace6fa1d819017584bdbcce462a8 \
  --output /tmp/new-neural-noise-candidate.json
```

See the [acquisition specification](metricgan-acquisition-v1.json) and retained
SpeechBrain source license. Experiments save durable prior-use identities, retain
original source grouping, and never overwrite earlier results. Labels are
publisher supplied/inherited; training exposure and independent source identity
remain unresolved. Results cannot establish general performance.

## Neural candidate results: rejected

The [neural-candidate receipt](core-neural-noise-candidate-v1.json) records a
serious false-alert failure:

| Condition/method | Raw AI detected | Raw human false alerts | Wrong definitive AI accusations |
| --- | ---: | ---: | ---: |
| Clean, unchanged | 25/25 | 0/20 | 0 |
| Clean, MetricGAN | 25/25 | 13/20 | 8 |
| Added noise, unchanged | 19/25 | 1/20 | 0 |
| Added noise, MetricGAN | 25/25 | 20/20 | 14 |

Recovering all noisy synthetic clips is not an improvement when every noisy
human clip is also falsely flagged at the raw threshold. The existing guards
abstain on some, but 14 noisy genuine clips still receive wrong definitive
synthetic indications after enhancement. The neural candidate is rejected.
The experiment measures changed classifier behavior, not the mechanism causing
those false alerts or independently judged enhancement quality.

All three preprocessing candidates are now closed for this round. None is
activated in the app. The current human-corroboration safeguard remains intact;
no classifier score, weight or threshold changed. The evidence points away from
these cleanup methods as a solution and toward a detector trained for broader
noise/generator conditions, with recalibration and fresh evaluation. Further
parameter searches on this small cohort would risk optimizing to the examples
already inspected.

## Verification

Local automated checks cover deterministic timing/length, zero-input behavior,
analytic low-pass response, invalid input/checkpoint rejection and durable
prior-use registration. Six actual-weight upstream tensor parity fixtures passed
with zero maximum absolute error. The complete local suite passed 267 Python
tests; no hosted CI or independent replication is claimed.
