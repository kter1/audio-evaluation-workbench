# Interaction Research Lab

**Local-first annotation QA, audio research evidence, and forensic investigation.**

This is the new portfolio-facing workflow in the same repository as the original
Audio Review Lab. The existing annotation study, source-code detectors and frozen
evaluation receipts are retained; the personal listening challenge is not required.

## Quickstart

Run \`./run.sh\` and select **Load demonstration dataset**. The app will
create a local SQLite database in the git-ignored \`data/\` directory, import
constructed metadata/annotation examples, and run deterministic checks.
Explore the **Investigations** queue and export the report under **Reports**.

For the original human annotation study and optional full local detector
inference, run \`./run-legacy.sh\`. Large model checkpoints are not required for
the new app. Do not treat the sample fixture as genuine research evidence.

## Workflow

1. **Import:** JSON manifest containing recording rows and annotation rows.
   See [research schema](docs/research-schema.md). Imports are immutable and
   idempotent; a changed manifest needs a new dataset ID.
2. **Audit:** explicit, versioned rules flag objective defects (invalid label,
   duplicate ID, bad timespan) or review candidates (conflicting human labels).
3. **Investigate:** inspect source metadata and imported annotations, add a
   status and rationale, and retain an append-only disposition history.
4. **Forensics:** inspect prior real detector evaluation receipts and optionally
   submit local audio for descriptive analysis. The original app supports
   PELLA v2 + AntiDeepfake fusion and signal diagnostics.
5. **Report:** export summary JSON, issues CSV, dispositions CSV, and HTML.
   Original audio, credentials and local media paths are never bundled.

## Demonstration credibility

The included demo contains **deliberately broken, constructed metadata**.
The issues it produces are seeded software tests; counts are not estimates of
real-world anomaly rates or detection accuracy. No Pindrop interaction data or
audio content is included.

The historical deepfake benchmark uses 320 frozen source clips in four simulated
channel conditions; see [evaluation methodology](docs/model-evaluation.md).
Those prior test results are not newly scored by the no-model quickstart.
The default fusion includes noncommercial research weights. Do not distribute
or claim commercial deployment rights without reviewing source licenses.

## Verification

Lightweight QA:
\`\`\`sh
python -m pytest -q tests/test_research.py tests/test_research_ui.py
\`\`\`
A CI workflow runs the new research suite without downloading model weights.
The original full suite remains runnable with the locked requirements.

## Portfolio walkthrough

1. Show fixture provenance and explain that defects are planted.
2. Load the demo dataset; inspect the top rules and flagged cases.
3. Open a conflicting-label finding; show why it is a **review candidate**,
   not an automated declaration of human error.
4. Record a case disposition and show the immutable event history.
5. Export research findings and inspect the limited historical detector
   evaluation receipt.
