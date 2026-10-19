"""Adapters for the actual ASVspoof 2021 LA files (not guessed CSV formats)."""
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile
import urllib.request

from .audio import checksum

KEYS_URL = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
MAPS_URL = "https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz"
DATA_URL = "https://zenodo.org/records/4837263"


def source_group(row):
    for field in ("VCTK_ID", "VC_source_VCTK_ID"):
        if row.get(field, "-").strip() not in ("", "-"):
            # Group genuine speech and voice conversions of the same source together.
            value = re.sub(r"_mic[12]$", "", row[field].strip())
            return "vctk:" + value
    text = row.get("TTS_text", "-").strip()
    if text not in ("", "-"):
        normalized = " ".join(re.findall(r"\w+", text.casefold()))
        return "tts-text:" + hashlib.sha256(normalized.encode()).hexdigest() if normalized else None
    return None


def parse_mapping(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"ASVspoof_ID", "VCTK_ID", "TTS_text", "VC_source_VCTK_ID"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Use ASVspoof2021_LA_VCTK_MetaInfo.tsv from the official mapping archive")
        for row in reader:
            trial = row["ASVspoof_ID"].strip()
            if trial in result:
                raise ValueError("Duplicate source mapping ID")
            result[trial] = source_group(row)
    return result


def import_asvspoof(store, metadata, scores, mapping, audio_dir):
    metadata, scores, mapping = map(Path, (metadata, scores, mapping))
    source = parse_mapping(mapping)
    records = []
    with metadata.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 8:
                raise ValueError(f"LA metadata line {number}: expected eight whitespace-separated fields")
            speaker, trial, codec, transmission, attack, reference, trim, subset = fields
            if not re.fullmatch(r"LA_E_\d+", trial) or reference not in ("bonafide", "spoof") or subset not in ("eval", "progress", "hidden"):
                raise ValueError(f"Invalid official metadata at line {number}")
            records.append({"trial_id": trial, "path": str(Path(audio_dir) / (trial + ".flac")),
                            "source_group": source.get(trial), "subset": subset, "codec": codec,
                            "attack": attack, "reference": "human" if reference == "bonafide" else "synthetic", "kind": "research"})
    score_rows = []
    with scores.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"LFCC-GMM scores line {number}: expected trial ID and score")
            score_rows.append({"trial_id": fields[0], "score": float(fields[1])})
    if not records:
        raise ValueError("No recordings in metadata")
    unmapped_progress = [r for r in records if r["subset"] == "progress" and not r["source_group"]]
    if unmapped_progress:
        raise ValueError("All progress recordings need source mappings to prevent calibration overlap")
    provenance = {"dataset": "ASVspoof 2021 Logical Access", "dataset_url": DATA_URL,
                  "metadata_url": KEYS_URL, "mapping_url": MAPS_URL, "baseline": "published LFCC-GMM",
                  "metadata_sha256": checksum(metadata), "scores_sha256": checksum(scores),
                  "mapping_sha256": checksum(mapping), "license": "ODC-By 1.0; retain official notices",
                  "grouping": "VCTK source ID for human/VC; normalized input text for TTS. Cross-type text identity is not fully observable.",
                  "records": len(records), "scores": len(score_rows),
                  "unmapped_eval_excluded": sum(r["subset"] == "eval" and not r["source_group"] for r in records)}
    store.register(records, score_rows, provenance)
    return provenance


def fetch_metadata(destination, keys_archive=None, maps_archive=None):
    """Download metadata only, with known members; never extract arbitrary paths."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archives = []
    for local, url, name in ((keys_archive, KEYS_URL, "LA-keys-full.tar.gz"),
                             (maps_archive, MAPS_URL, "source-maps.tar.gz")):
        path = Path(local) if local else destination / name
        if not path.exists():
            if local:
                raise ValueError(f"Archive does not exist: {path}")
            temporary = path.with_suffix(".partial")
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        archives.append(path)
    with archives[0].open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != "f052cc2ed276745afa3b5198665d3b26":
        raise ValueError("Official LA key archive checksum mismatch")
    members = ((archives[0], {"keys/LA/CM/trial_metadata.txt": "trial_metadata.txt",
                              "keys/LA/CM/LFCC-GMM/score.txt": "lfcc_gmm_scores.txt",
                              "keys/LA/README.txt": "keys_README.txt"}),
               (archives[1], {"ASVspoof2021_LA_VCTK_MetaInfo.tsv": "source_mapping.tsv",
                              "LICENSE.txt": "mapping_LICENSE.txt", "README.txt": "mapping_README.txt"}))
    for archive, names in members:
        with tarfile.open(archive) as tar:
            for member, name in names.items():
                entry = tar.getmember(member)
                if not entry.isfile() or entry.size > 150_000_000:
                    raise ValueError("Unexpected metadata archive member")
                (destination / name).write_bytes(tar.extractfile(entry).read())
    result = {"keys_url": KEYS_URL, "mapping_url": MAPS_URL,
              "keys_archive_sha256": checksum(archives[0]), "mapping_archive_sha256": checksum(archives[1])}
    (destination / "download_provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
