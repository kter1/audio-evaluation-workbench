# Research manifest format (version 1)

The JSON manifest is immutable under its dataset_id. Reimporting identical bytes is idempotent;
a changed manifest requires a new versioned dataset_id. Entries may contain intentionally
bad data; the auditor flags record-level problems instead of silently fixing them.

Required top-level keys: dataset_id, name, kind, origin, recordings[], annotations[].
Each recording should contain recording_id and duration_seconds (>0 finite).
Optional: channel, codec, media_path (relative to an explicitly selected media root),
sha256 (64 hex), reference_label, interaction_id, source provenance.
Each annotation should contain annotation_id, recording_id, source, label
(human/synthetic/uncertain), guideline_version; optional start_seconds,
end_seconds, escalated (boolean), escalation_reason.

Reference labels are dataset-supplied claims, not model measurements. Annotations
are reported as imported human/mocked source labels. Model predictions must be stored
separately and cannot substitute for a trusted reference.

This release accepts a JSON metadata manifest and optional media root, not arbitrary
archives. Paths that escape the media root are rejected, and no URL or remote media is
fetched. Media is never copied into the repository. Refrain from importing personal
or proprietary recordings unless permitted.

The included demonstration is labeled synthetic metadata, and referenced audio
is intentionally missing to exercise quality controls. It is not a deepfake benchmark.
See docs/model-evaluation.md for prior real-data evaluation results and limits.
