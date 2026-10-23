# Complementary detector calibration probe

A deterministic SHA-256 split assigned 36 connected source groups to calibration
and 37 to evaluation; every recording, reference and condition stayed within its
group. This is a retrospective split of previously inspected v4 regression, not
an untouched test. No claim of independent model selection is made.

The fixed rule chooses the next floating-point score above the maximum genuine
calibration score across all four conditions. Synthetic labels and evaluation
scores do not choose the threshold. The resulting native WavLM threshold is
0.33933633565902716. It is research-only; app thresholds remain frozen.

Evaluation contains 100 genuine and 53 synthetic originals, reused across four
conditions. Clean, MP3 and telephone conditions each detect 53/53 synthetic
recordings with 0/100 genuine alerts. Added noise detects 52/53 with 1/100 genuine
alerts. Calibration detects 63/67 noisy synthetic recordings: the operating point
still misses AI speech. The calibration group's zero-error one-sided 95% upper
bound is about 8%, so these counts cannot substantiate a 2% population-risk claim.

Do not promote this threshold. It adds a noisy-human alert relative to native
0.5 on these sources, and partial-synthesis limitations remain unresolved.
Next work requires source-diverse genuinely untouched calibration/evaluation,
not further threshold searches on this split. No weights or app defaults changed.

[Complete receipt](complementary-calibration-v1.json).

Verification: two tests pass, including an evaluation-score mutation test proving
that evaluation scores cannot affect the selected threshold, plus duplicate/NaN
rejection. Existing historical reports remain intact.
