# Detector method · version 4.0

## What the instrument does

The lab performs actual local inference on supplied speech, then reports a model prediction, temporal scores, signal measurements, quality limitations, and controlled perturbation results. Known reference provenance is displayed separately from predictions. A short input is identified before inference; the lab can still show its raw model vote while marking its assessment inconclusive. A score is not proof of audio origin.

## Default detector: two-model fusion

The default combines two independently trained speech-deepfake detectors, window by window:

- **[pellav2](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector/tree/a4c68a3d4afc7f5d9450c94f5d0652aaa208d40a)** (MIT): an XLS-R-300M backbone with learned layer weighting, trained by its authors on modern TTS. On [Podonos's independent benchmark](https://www.podonos.com/blog/audio-deepfake-detection-benchmark) of about 25 modern voice systems it was the strongest downloadable model. In this lab it catches modern fakes reliably but mistakes compressed or noisy real speech for synthetic.
- **[NII AntiDeepfake MMS-300M](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake/tree/7928e119e09f616b2ce698892e00ca22dbaa40bf)** ([paper](https://arxiv.org/abs/2506.21090); weights CC BY-NC-SA 4.0, non-commercial): post-trained on about 56,000 hours of real and 18,000 hours of artifacted speech in 100+ languages, with augmentation. It is far more robust on compressed real speech, but misses some fakes on its own.

**Fusion.** For each 4-second window, z = 2.0425 + 0.3069·a + 0.9340·b, where *a* is pellav2's logit and *b* is AntiDeepfake's fake-minus-real logit (both clipped to ±30). The window probability is sigmoid(z), and the file score is the mean over windows. The three weights and the 0.2096 threshold were fitted on development data only (the already-reported v3 clips in clean, MP3, telephone, and noisy versions) under a rule fixed in advance; see [the evaluation](model-evaluation.md). The combined detector therefore inherits AntiDeepfake's **non-commercial** license.

**Ports and parity.** pellav2 is vendored from its upstream inference script with only the backbone construction changed; `scripts/check_pellav2_parity.py` matched upstream exactly (largest difference 0.0 on five clips). AntiDeepfake ships as fairseq code, so the lab maps its checkpoint keys onto the identical `transformers` Wav2Vec2 architecture and drops the unused pre-training quantizer. `scripts/check_antideepfake_parity.py` runs the official model-card code in an isolated Python 3.9 environment with fairseq at the authors' pinned commit; on identical 16 kHz input the largest logit difference was 9.2e-5 across eight clips. The official script resamples with torchaudio and the lab with SciPy, so non-16 kHz files can score slightly differently from the upstream script. All weights load through restricted loaders (`weights_only=True`, safetensors) with pinned SHA-256 hashes and strict key matching.

pellav2, AntiDeepfake, the 2019-era [AASIST](https://github.com/clovaai/aasist/tree/a04c9863f63d44471dde8a6abcb3b082b07cd1d1), and the earlier [Wav2Vec2-XLSR](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector/tree/c66306024a7ede0be291e9c4558b37634782dc4e) remain individually selectable. Reports always store logits in human, synthetic order. Parameter count does not establish detector quality.

## Input and preprocessing

Limits: 50 MiB, 0.5–120 seconds, 1–8 channels, 8–192 kHz, ≤24 million decoded samples. Decode errors, mismatched frame counts, and nonfinite samples stop processing. The strict legacy FLAC fallback accepts only PCM matching the original embedded checksum. Input SHA-256 identifies exact bytes. Processing is local; the input audio is not submitted to an inference service.

All detectors use arithmetic channel-mean mono and 16 kHz polyphase resampling (SciPy `resample_poly`, reduced rational factors, default Kaiser filter). There is no automatic denoising or silence trimming. All run on CPU float32 in eval/inference mode.

The fusion and both of its models use 64,000-sample (4-second) windows. AntiDeepfake layer-normalizes each window and scores it without padding; its authors score whole utterances, so 4-second windows are this lab's documented choice. pellav2 uses the same windows, the length of the upstream crop. Each window is zero-padded at the end if shorter, then standardized to zero mean and unit variance, in that order, as upstream. Longer recordings use complete windows plus a final window anchored at the file end, and the file score is the arithmetic mean of window scores. Upstream scores a single centre crop instead; covering the whole file is this lab's documented choice. Because each window is standardized, the score does not depend on input gain.

AASIST uses 64,600-sample (4.0375-second) windows. A recording shorter than one window repeats its waveform to that length, matching the authors' `pad` behavior. Longer recordings use complete windows; if a tail remains, a complete window anchored at the file end includes it. This final window may overlap the preceding window. **Tiny tails are never independently repeated and scored.** Every source sample is covered. File score is the arithmetic mean of contextual window scores. Overlapping scores are correlated; this is a declared aggregation choice, not a calibrated fusion model. No amplitude normalization is applied.

Wav2Vec2 uses consecutive 6-second windows; a final tail shorter than 2.5 seconds merges into its preceding window. Its feature extractor normalizes each window. Its file score is the duration-weighted mean of non-overlapping window softmax outputs. No zero padding is used. Both pipelines retain original time bounds and both logits in reports. Neither windowing method localizes edited boundaries precisely.

## Reliability policy

The fixed engineering policy abstains when the aggregate score falls within 0.15 of the detector's binary threshold (0.35–0.65 at 0.5), or, for the fusion, within ±2 on the logit scale around its 0.2096 threshold (0.035–0.66), when windows fall both below and above that band, when input is shorter than 2.5 seconds, when mono RMS is below −50 dBFS or energy activity below 10%, when more than 1% of samples clip, or when stereo correlation is below −0.5. These rules are heuristics, not trained confidence intervals, and have not been relaxed to improve demos.

Every report opens with **Why this result**: the vote (aggregate score against the threshold), each model's own score before fusion, and each reliability check with its measured value, limit, and pass/fail. A shift larger than 0.20 under a sensitivity probe is listed as a caution but does not by itself make the assessment inconclusive. The raw vote and the qualified assessment are always shown separately. Speech presence and language are not independently verified. Music, silence, other languages, unseen synthesis systems, and editing can still cause confidently wrong outputs.

## Signal laboratory

Waveform plots show minima/maxima over sample blocks. The STFT uses a 25 ms Hann window, 10 ms hop, 512-point FFT, and no boundary extension. Color is magnitude dB relative to the file maximum, with ≤1200 max-pooled display columns. Silence displays at the floor. Welch PSD uses a 512-sample Hann window and 256-sample overlap.

RMS/peak are relative to amplitude 1.0. Crest factor is peak/RMS in dB; DC is the mean. Clipping counts original channel samples with |x|≥0.999. Activity counts 20 ms frames above −50 dBFS, excluding an incomplete last frame; it is energy activity, not VAD. The lab also reports per-channel RMS, stereo correlation, PSD-weighted spectral centroid, 95% rolloff, geometric/arithmetic PSD flatness, sign-change rate, and power fractions in 0–300, 300–3400, and 3400–8000 Hz bands.

These measurements and plots are not saliency maps or independent evidence of AI origin. Compression and recording conditions also affect them.

## Sensitivity testing

The most energetic inference window is tested with three independent transformations: half gain (−6.0206 dB), sixth-order Butterworth low-pass at 3400 Hz applied forward/backward, and Gaussian noise at 30 dB RMS SNR with NumPy RNG seed `20261019`. Every transformed score is compared with the same unmodified window. A shift >0.20 creates a visible warning. Transformations are never combined. Stability is distinct from correctness; these are not independent detector votes or adversarial certification.

## Artifacts and reproduction

Install the default with `scripts/setup_pellav2.py` and `scripts/setup_antideepfake.py` (about 1.26 GB each, verified against pinned hashes; the second prints its non-commercial license); the fusion weights are tracked in `lab/fusion-v4.json`. `scripts/setup_aasist.py` and `scripts/setup_detector.py` install the comparison models. Dependencies are pinned in `requirements.lock`. Subsequent inference requires no network. Use `scripts/analyze_audio.py recording.wav --robustness`; add `--detector pellav2`, `--detector antideepfake`, `--detector aasist`, or `--detector wav2vec2` for a single model.

Immutable content-addressed JSON reports are saved under `data/analyses`; JSON and printable HTML are downloadable. Reports record source checksums, model/code/config hashes, exact preprocessing, windowing, aggregation, threshold, class scores/logits, reliability rules and their outcomes, signal measurements, sensitivity results, versions, and timing. Absolute timing and timestamps vary; the pinned CPU outputs are checked for repeatability. Uploaded audio stays in the app session; no forensic operation creates personal study judgments.
