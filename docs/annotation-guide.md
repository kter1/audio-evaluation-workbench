# Speech annotation guide · version 1.1

## Purpose

Describe what you hear consistently. A listening impression does not establish a recording's origin, a person's identity, or fraud. Do not use an external detector, search the recording, inspect filenames, or consult reference labels during blind review. Detector output is compared only after a batch is revealed.

Practice fixtures are generated tones with arbitrary test labels. They exercise software only and are excluded from all reported results.

## Review sequence

1. Listen once at the original speed and comfortable volume. Replay difficult intervals at the original sample rate; avoid changing pitch, denoising, or otherwise altering evidence.
2. Record intelligibility and interference. Add timestamped observations when you hear a specific anomaly. Timestamps are seconds from the original recording, including when you listen to an extracted segment.
3. Mark the perceptual cues you actually heard, if any.
4. Assign perceived origin, confidence, and reasoning. Use **uncertain** when the evidence does not support a defensible origin judgment.
5. Escalate issues outside the guide, conflicting evidence, or a case requiring additional information. Explain the question someone else should investigate.
6. Save a draft or submit. Submission freezes the original judgment; later corrections and adjudications are separate records.

## Intelligibility (one category)

| Label | Definition | Example / exclusion |
|---|---|---|
| clear | Most words are understandable without unusual effort | A quiet accent alone is not degradation. |
| partly intelligible | Some speech is understandable, but a material portion is obscured | A dropout removes words; do not transcribe invented words. |
| unintelligible | Speech is audible, but you cannot reliably understand it | An unfamiliar language is not proof of poor signal quality; explain the language limitation. |
| no speech | No speech is audible | Silence, music, or a test tone; this is not a human/synthetic origin label. |

## Interference (zero or more tags)

Interference describes the recording channel and environment. None of these tags is evidence of synthesis by itself.

- **background noise:** audible sound competing with the speech, such as traffic, hum, or room noise.
- **compression artifacts:** watery, swirling, or band-limited sound typical of low-bitrate audio or phone codecs. Common in genuine calls.
- **reverb or echo:** audible room reflections or a delayed repeat of the voice.
- **music or hold audio:** music, hold tones, or IVR prompts mixed with or replacing the speech.
- **clipping:** audible harsh saturation, supported where useful by flattened waveform peaks. Do not infer clipping from loudness alone.
- **dropouts:** a brief loss or abrupt interruption in an otherwise continuous signal. Ordinary pauses are excluded.
- **distortion:** audible warbling, metallic texture, or other alteration not explained by the tags above.
- **overlap:** simultaneous voices that affect review. Do not infer how many identities are present.
- **other:** describe the specific observation; do not invent a new undisclosed classification rule.

## Perceptual cues (zero or more)

Cues record *why* a clip sounds the way it does to you. Mark only cues you actually heard, before reveal. After reveal, Findings shows how often each cue coincided with synthetic speech, so unhelpful cues can be dropped from future versions. A cue is evidence, never a verdict: genuine speech can sound flat, and modern synthesis can breathe.

Leaning synthetic:

- **unnatural prosody or rhythm:** stress, pitch, or timing that does not fit the meaning, such as even pacing through a list or a question without a question's melody.
- **flat or uniform emotion:** the same energy and tone across content that would naturally vary.
- **metallic or robotic timbre:** buzzy, phasey, or vocoder-like voice quality not explained by the channel.
- **missing breaths or mouth sounds:** long fluent stretches with no audible breaths, lip or tongue sounds where a reader would normally produce them.
- **odd pronunciation or stress:** a word stressed or pronounced in a way a fluent speaker of that accent would not, especially names and rare words.
- **abrupt splice or transition:** a sudden change in voice, room, or level mid-utterance.
- **too-clean studio sound:** no room tone at all, in a context where some would be expected.

Leaning human:

- **natural breaths or mouth sounds:** breaths placed at natural phrase boundaries.
- **natural hesitation or disfluency:** restarts, fillers, or self-corrections that fit the content.
- **consistent room acoustics:** one stable acoustic space throughout, including its noise.

- **other cue:** describe it in your decision notes.

## Perceived origin

- **human:** the recording sounds consistent with human speech to you. This is a perceptual judgment, not verified origin.
- **synthetic:** you hear evidence you judge consistent with generated or converted speech. Describe the evidence; one cue alone is rarely conclusive.
- **uncertain:** evidence is insufficient, conflicting, non-speech, or outside your expertise. Explain what prevents a decision. Uncertainty is a valid outcome, not an error to conceal.

## Confidence

1 = very weak evidence; 2 = limited evidence; 3 = moderate evidence; 4 = strong evidence; 5 = very strong evidence. Confidence concerns your judgment and is not a calibrated probability. High confidence does not override a reference label.

## Edge cases

- **Phone or compressed audio:** codecs remove high frequencies and add artifacts to genuine speech. Tag the channel; do not treat band-limited sound as a synthesis cue.
- **Background noise:** noise can hide breaths and room tone. Prefer uncertain over guessing when noise masks the cues you would rely on.
- **Very short clips (under about 3 seconds):** judge cautiously and say that duration limited the evidence.
- **Read or scripted speech:** audiobook and script reading is naturally even and fluent. Expect fewer disfluencies; do not count their absence as a synthetic cue on its own.
- **Accents and non-native speech:** unfamiliar stress patterns are not "odd pronunciation" unless they are inconsistent within the same speaker.
- **Emotional or performed speech:** exaggerated delivery is not unnatural prosody by itself.
- **Hold music, IVR prompts, or recorded announcements:** recorded prompts are machine-played but not deepfakes. Tag music or hold audio and judge only the caller's speech; escalate if you cannot separate them.

## Observations and escalation

Write a reproducible observation such as “1.18–1.36 s: abrupt silence while the word appears interrupted.” Avoid “the model is wrong because this is fraud.” Start must precede end and both must fall inside the recording. An uncertainty explanation and an escalation reason are mandatory when applicable.

Escalate language limitations, a guideline gap, contradictory evidence, or a suspected data issue. Corrupt or missing media must be repaired before submission; never label it as a completed review.

## Quality review

Blind repeat reviews occur at least 48 hours after the original submission. Earlier answers stay hidden. Independent reviewers use their own workspace without access to primary judgments. Compare original submissions first; record adjudication separately.

For a revised guideline, register a new immutable version before starting a fresh batch. Document exactly what changed and why. Different batches allow descriptive comparison, not proof that the revision caused improvement.

## Changes

- **1.1:** added perceptual cues (recorded before reveal and analyzed afterwards); added compression artifacts, reverb or echo, and music or hold audio as interference tags; added the edge-case section. Registered before any review began.
- **1.0:** initial guide.
