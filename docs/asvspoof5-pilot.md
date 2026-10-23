# Unscored external-audio pilot

Acquired40 ASVspoof5 evaluation recordings from the official Zenodo release:
20 genuine and20 synthetic recordings from40 distinct declared speakers.
Source: https://zenodo.org/records/14498691 . ASVspoof5 organizing committee,
2024. Contains information from ASVspoof5 made available under the ODC Attribution
License; bona fide contents are CC BY4.0 per the official README. Original notices
are retained beside local audio.

The acquisition read only the first8MiB of each of the ten official evaluation
TARs:80MiB total audio transfer, not the142GB dataset. It found678 complete
protocol-listed members and chose40 via a fixed SHA-256 metadata rule before
audio decoding or inference. Selected audio totals4,845,581bytes and spans
3.44–10.46seconds, thirteen synthetic attack labels and ten codec labels plus
uncompressed audio. This prefix-based pilot is not a representative benchmark.

[Acquisition receipt](asvspoof5-pilot-acquisition-v1.json) records official
protocol membership, byte ranges, archive offsets, member SHA-256 hashes,
publisher archive checksums and license notices. Full-archive publisher MD5
cannot be verified from prefixes; that limitation is retained explicitly.

[Integrity audit](asvspoof5-pilot-qualification-v1.json) verified all ten cached
prefix hashes, reconstructed every selected complete member, decoded all40
recordings, checked finite mono16k audio, rejected source/decoded duplicates,
and found zero source-byte overlaps with registered project history.
All selected codec-parent references resolve in the official protocol to the
same declared speaker. Neither this check nor distinct anonymized IDs establishes
biometric identity independence from older corpora.

## Reservations and remaining qualification

The first group-only hash split happened to contain6genuine and14synthetic
calibration recordings. Before any inference, a metadata-only stratified update
reserved10genuine+10synthetic recordings per partition, with no declared speaker
crossing partitions. The original acquisition/qualification receipts are preserved;
[reservation version2](asvspoof5-pilot-reservation-v2.json) is authoritative for
future pilot work. No score or audio-quality result selected these partitions.

**No detector has scored this pilot. No app weights or thresholds changed.**

These data are new to the registered project acquisition history, not certified
absent from upstream model training. Cross-corpus perceptual duplicates, speaker
identity overlap and training exposure remain unresolved. Counts are well below
the confirmatory300-genuine-group gate. Do not calibrate or promote a production
threshold from this pilot, and do not call its reserved evaluation a qualified
independent locked set. Expand and qualify provenance-backed groups before
threshold selection or final evaluation.

Checks:14 focused tests passed, including strict byte-range enforcement,
truncated/unsafe TAR handling, deterministic metadata selection, partition
disjointness and existing acquisition/calibration safeguards. Compilation and
Git diff checks passed. The full decoded-data integrity audit also passed.

Recheck local integrity using a new audit output version if adapting the audit
script; existing evidence outputs intentionally refuse overwrite. Research
sources are under data/research-corpora/asvspoof5-pilot and remain outside Git.
