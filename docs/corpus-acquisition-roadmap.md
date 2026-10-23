# Evaluation corpus inventory and acquisition roadmap

## Local inventory

Reproduce without downloads or inference:

```sh
.venv/bin/python scripts/inventory_corpora.py
# Optional new output path; existing evidence is never overwritten.
.venv/bin/python scripts/inventory_corpora.py --output /tmp/new-corpus-inventory.json
```

[Inventory receipt v1](corpus-inventory-v1.json) covers four historical manifests:
**840 case rows, 800 unique original byte hashes and 187 connected identity groups**.
Repeated cases across historical manifests are counted once in the unique-byte
total. Derivative codecs are not counted as independent originals.

Grouping connects byte identity, source utterance, cloning reference, target
speaker and available source speaker transitively. LibriSpeech and VCTK speaker
namespaces are separate. The ASVspoof source mapping supplies VCTK identities;
all manifested rows have at least one resolved speaker identity. Speaker-ID
resolution is not biometric verification or a proof that the metadata is complete.

| Manifest | Case rows | Connected groups crossing its historical splits |
| --- | ---: | ---: |
| ASVspoof v1 | 80 | 18 |
| ASVspoof corrected v2 | 80 | 19 |
| Modern voices v3 | 360 | 0 |
| Fusion evaluation v4 | 320 | 0 |

The ASVspoof selection/test split was source-group-disjoint, but speaker identity
connects cases across those splits. It does not meet the new connected-speaker
independence requirement. The receipt lists group IDs and member record IDs so
the finding is reviewable. These are historical limitations, not newly measured
changes in detector accuracy. No historical manifest or partition was rewritten.
The previously audited modern-development/v4-evaluation boundary remains separate.

The inventory also finds 464 extracted ASVspoof audio files, 11,126 LibriSpeech
files, 208 generated modern-voice files and 148 generated v4 files in the scanned
source directories. File counts include unmanifested material; they establish
neither untouched status nor usable independent-group counts. File existence and
manifest identities are checked here; content hashes for the v4 baseline are
verified by the separate baseline audit.

**Zero local groups have been qualified as a newly locked test.** Historical
results have already been inspected. Raw local audio needs a provenance,
previous-use, speaker/source/reference and perceptual-duplicate audit before any
new partition could be proposed.

## External acquisition sequence

Candidate resources below were checked against primary publisher pages. None
was downloaded, partitioned or scored. Public benchmarks can support external
comparison, but public availability cannot establish absence from detector training.

1. **ASVspoof 5 — broader independent comparison.** The official release provides
   training/development/evaluation protocols and audio, with about 2,000 speakers
   and varied recording conditions. Its full release is listed at 142.3 GB.
   First inspect protocols and model-training exposure; choose required shards
   before budgeting storage. Use the official evaluation protocol for published
   comparisons; identify any locally grouped/subsampled analysis separately.
   Sources: [official release](https://zenodo.org/records/14498691),
   [README](https://zenodo.org/records/14498691/files/README.txt?download=1),
   [license](https://zenodo.org/records/14498691/files/LICENSE.txt?download=1).

2. **In-the-Wild — external domain stress test.** The publisher reports 58 public
   figures with both genuine and synthetic audio. Its speaker count cannot meet
   this contract's 300 independent human groups for a confirmatory condition.
   Treat it as a disclosed external stress test rather than relaxing that gate.
   Source: [publisher dataset page](https://deepfake-total.com/in_the_wild).

3. **MLAAD plus corresponding genuine speech — multilingual/generator shift.**
   MLAAD provides synthetic speech; its publisher points to M-AILABS for genuine
   speech and lists a non-commercial license. Plan class-matched domains,
   languages and recording conditions to avoid learning dataset identity instead
   of synthesis. Resolve generator-family relationships and upstream training
   exposure before choosing held-out families.
   Source: [publisher dataset page](https://deepfake-total.com/mlaad).

4. **ReplayDF — recorded replay conditions.** The publisher describes replayed
   M-AILABS/MLAAD material across 109 speaker–microphone combinations. These are
   device combinations, not 109 independent human speakers. Connect each replay
   to its original and keep all such derivatives in the same partition, including
   when MLAAD material appears in another corpus. The publisher specifies
   non-commercial research use.
   Source: [publisher dataset page](https://deepfake-total.com/replaydf).

No one corpus covers all contract conditions. Naturally recorded device/noise
variation, short speech, partially synthetic speech and provenance-backed edit
boundaries need separately documented coverage. Controlled edits can validate
known transformations but do not substitute for unknown real-world manipulations.

## Before acquisition or scoring

- Record dataset version, selected files, checksums, provenance, permitted use,
  compressed/extracted storage budget, annotation availability and training exposure.
- Maintain a registry of all previously used source identities and audio hashes.
- Build connected identity groups across the entire proposed corpus, not independently
  per dataset. Audit perceptual duplicates and document unresolved metadata.
- Prespecify supported domains, sample minimums, metrics and condition weights.
- Reserve development, calibration and locked groups without consulting detector scores.
- Do not expose locked labels/results to candidate tuning. Freeze model revisions,
  score aggregation, operating thresholds and abstention before final scoring.

The next data milestone is a qualified metadata manifest and storage estimate,
not an automatic large download or a claim of independent validation.

## Acquisition update

A bounded 30-clip TEN development set and ASVspoof 5 protocol metadata have now
been acquired and verified. No ASVspoof 5 audio was acquired. TEN identities remain
unresolved and its clips have been consumed as development evidence. Neither
source has passed prospective qualification. See the
[VAD development study](vad-development-study.md) for results and limitations.
