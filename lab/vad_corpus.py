"""Strict import of publisher binary speech annotations, without inventing labels."""
import csv
from hashlib import sha256
import math
from pathlib import Path

import soundfile as sf


def parse_binary_intervals(text, recording_id, duration):
    rows = list(csv.reader(text.splitlines()))
    if len(rows) != 1 or not rows[0] or rows[0][0] != recording_id or (len(rows[0])-1) % 3:
        raise ValueError("Expected one matching recording ID followed by start,end,binary-label triples")
    fields = rows[0][1:]
    if not fields:
        raise ValueError("Empty speech annotation is unknown, not a negative reference")
    speech, previous = [], 0.
    for i in range(0, len(fields), 3):
        start, end = float(fields[i]), float(fields[i+1])
        label = fields[i+2]
        if (not math.isfinite(start) or not math.isfinite(end) or label not in ("0", "1")
                or start < 0 or start >= end or abs(start-previous) > 1e-8 or end > duration+.0011):
            raise ValueError("Speech labels must be finite, contiguous, binary and within annotation precision of the source")
        previous = end
        end = min(end, duration)
        if label == "1" and start < end:
            if speech and abs(speech[-1]["end"]-start) < 1e-8:
                speech[-1]["end"] = end
            else:
                speech.append({"start": start, "end": end})
    if abs(previous-duration) > .0011:
        raise ValueError("Annotation does not cover the complete recording within 1 ms rounding precision")
    return speech


def ten_manifest(root, directory, revision, downloads):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    cases = []
    for path in sorted(directory.glob("testset-audio-*.wav")):
        annotation = path.with_suffix(".scv")
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError("Expected source mono 16 kHz in this pinned corpus")
        intervals = parse_binary_intervals(annotation.read_text(), path.stem, info.duration)
        cases.append({"id": "ten-vad:"+path.stem,
            "source_group": "ten-vad:testset:unresolved-source-identities",
            "speaker_group": "ten-vad:testset:unresolved-source-identities",
            "source_references": [], "path": str(path.relative_to(root)),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "annotation_provenance": f"TEN-framework/ten-vad {revision}, testset/{annotation.name}; publisher reports manual annotations; independent review pending",
            "annotation_sha256": sha256(annotation.read_bytes()).hexdigest(),
            "speech_intervals": intervals, "level_events": None, "noise_reference_dbfs": None})
    if len(cases) != 30:
        raise ValueError("Pinned TEN development corpus must contain exactly 30 complete recording/annotation pairs")
    return {"schema_version": "method-benchmark-1.0", "partition": "development",
            "source_repository": "https://github.com/TEN-framework/ten-vad", "source_revision": revision,
            "cases": cases, "download_receipts": downloads,
            "limitations": ["Publisher-supplied manual labels; no independent annotation review performed here.",
                "Original speaker and source recording identities are incomplete; all clips conservatively share one group.",
                "Publisher mentions LibriSpeech, GigaSpeech and DNS Challenge sources; upstream training/local-history overlap is unresolved.",
                "No human/synthetic authenticity labels or edit/noise-reference annotations are inferred.",
                "Repository license includes additional conditions; no commercial redistribution or deployment clearance is asserted.",
                "Development stress test only; not a newly qualified locked corpus."]}
