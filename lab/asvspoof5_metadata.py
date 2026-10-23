"""Protocol-only coverage audit. No audio acquisition, inference or split reassignment."""
from collections import Counter, defaultdict
from hashlib import md5, sha256
import re
import tarfile

PROTOCOLS = {
    "ASVspoof5.train.tsv": (182357, "cbfa08e2fab1423e828f102bc97a22f0"),
    "ASVspoof5.dev.track_1.tsv": (140950, "e949dee1091a49b04943c548b5cf9860"),
    "ASVspoof5.eval.track_1.tsv": (680774, "50c0591d45a282cb787a5a4614394f67"),
}


def summarize_protocol(lines):
    counts, speakers, ids, references = Counter(), defaultdict(set), {}, []
    codecs, codec_speakers, attacks = defaultdict(Counter), defaultdict(set), Counter()
    checksum, strong = md5(), sha256()
    for raw in lines:
        if len(raw) > 4096:
            raise ValueError("Unexpected protocol line length")
        checksum.update(raw)
        strong.update(raw)
        fields = raw.decode("utf-8").split()
        if len(fields) != 10:
            raise ValueError("Track-1 protocol must contain ten space-separated fields")
        speaker, recording, gender, codec, quality, parent, adaptation, attack, label, reserved = fields
        if (not re.fullmatch(r"[TDE]_\d{4}", speaker) or not re.fullmatch(r"[TDE]_\d{10}", recording)
                or recording in ids or label not in ("bonafide", "spoof")
                or (label == "bonafide") != (attack == "bonafide")
                or (codec != "-" and not re.fullmatch(r"C\d{2}", codec))
                or (parent != "-" and not re.fullmatch(r"[TDE]_\d{10}", parent))):
            raise ValueError("Invalid or duplicated protocol identity/label")
        ids[recording] = speaker
        counts[label] += 1
        speakers[label].add(speaker)
        codecs[codec][label] += 1
        codec_speakers[codec, label].add(speaker)
        attacks[attack] += 1
        if parent != "-":
            references.append((speaker, parent))
    if not ids:
        raise ValueError("Empty protocol")
    parents = {speaker: speaker for speaker in set(ids.values())}
    def find(speaker):
        while parents[speaker] != speaker:
            parents[speaker] = parents[parents[speaker]]
            speaker = parents[speaker]
        return speaker
    unresolved, cross_speaker = 0, 0
    for speaker, source_id in references:
        source_speaker = ids.get(source_id)
        if source_speaker is None:
            unresolved += 1
        else:
            if source_speaker != speaker:
                cross_speaker += 1
            parents[find(speaker)] = find(source_speaker)
    conditions = {}
    for codec in sorted(codecs):
        conditions[codec] = {"rows": dict(codecs[codec]), "declared_speaker_groups": {
            label: len({find(s) for s in codec_speakers[codec, label]}) for label in ("bonafide", "spoof")}}
    return {"rows": len(ids), "class_rows": dict(counts),
            "declared_speaker_ids": len(parents),
            "declared_connected_speaker_groups": len({find(s) for s in parents}),
            "speaker_groups_by_class": {label: len({find(s) for s in values}) for label, values in speakers.items()},
            "codec_conditions": conditions, "attack_label_rows": dict(sorted(attacks.items())),
            "codec_seed_references": len(references), "unresolved_codec_seed_references": unresolved,
            "cross_speaker_codec_links": cross_speaker, "member_md5": checksum.hexdigest(),
            "member_sha256": strong.hexdigest()}


def audit_archive(path, expected_archive_md5):
    raw = path.read_bytes()
    if len(raw) > 25*1024*1024 or md5(raw).hexdigest() != expected_archive_md5:
        raise ValueError("Archive differs from bounded publisher-verified protocol release")
    summaries = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name not in PROTOCOLS:
                continue
            if not member.isfile() or member.size > 64*1024*1024 or member.name in summaries:
                raise ValueError("Invalid or duplicate protocol member")
            with archive.extractfile(member) as stream:
                result = summarize_protocol(stream)
            expected_rows, expected_md5 = PROTOCOLS[member.name]
            if result["rows"] != expected_rows or result["member_md5"] != expected_md5:
                raise ValueError("Protocol member differs from published row count/checksum")
            summaries[member.name] = result
    if summaries.keys() != PROTOCOLS.keys():
        raise ValueError("Missing required official protocols")
    return {"schema_version": "asvspoof5-metadata-audit-1.0", "archive_bytes": len(raw),
            "archive_sha256": sha256(raw).hexdigest(), "publisher_archive_md5": expected_archive_md5,
            "protocols": summaries, "audio_downloaded": False, "detector_inference_performed": False,
            "qualified_locked_corpus": False,
            "limitations": ["Speaker counts are declared protocol identities, not independently verified physical-speaker counts.",
                "Codec seed links connect declared speakers where observable; unresolved ancestry must be investigated before partitioning.",
                "Attack labels are not generator-family provenance; label differences do not prove model-family independence.",
                "Protocol metadata has no reviewed speech-activity, noise-level or edit-boundary annotations.",
                "Upstream detector training exposure, perceptual duplicates and cross-corpus physical-speaker overlap remain unresolved.",
                "Metadata coverage alone cannot satisfy corpus qualification or performance gates."]}
