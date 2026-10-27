"""Local corpus inventory and transitive identity grouping; never infer test eligibility."""
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

from lab.baseline_audit import checked_path

MANIFESTS = ("manifest-v1.json", "manifest.json", "modern-manifest.json", "v4-manifest.json")


def connected_groups(records):
    """Connect records sharing ANY identity token, including transitive references.

    Callers must namespace identities by their source dataset. Missing identities
    do not connect unrelated records and must be reported separately.
    """
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Duplicate record IDs in identity graph")
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, row in enumerate(records):
        for token in row["identity_tokens"]:
            if token in owners:
                parents[find(i)] = find(owners[token])
            else:
                owners[token] = i
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row["record_id"])
    return sorted((sorted(group) for group in groups.values()), key=lambda g: g[0])


def inventory(root, include_records=False):
    root = Path(root).resolve()
    loaded, manifests, records = [], [], []
    errors = []
    for name in MANIFESTS:
        path = root / "data/evaluation" / name
        if not path.is_file():
            errors.append(f"Missing manifest: {name}")
            continue
        raw = path.read_bytes()
        loaded.append((name, json.loads(raw)))
        manifests.append({"path": f"data/evaluation/{name}", "file_sha256": sha256(raw).hexdigest()})
    wanted = {c["trial_id"] for _, d in loaded for c in d["cases"] if c.get("trial_id")}
    mapping = {}
    mapping_path = root / "data/asvspoof/source_mapping.tsv"
    if mapping_path.is_file():
        with mapping_path.open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["ASVspoof_ID"] in wanted:
                    mapping[row["ASVspoof_ID"]] = row
    for name, document in loaded:
        cases = document["cases"]
        summary = next(m for m in manifests if m["path"].endswith("/" + name))
        summary.update({"case_rows": len(cases), "labels": dict(Counter(c["label"] for c in cases)),
                        "historical_splits": dict(Counter(c["split"] for c in cases)),
                        "eligible_as_locked_test": False, "reason": "Historical selected/published corpus; not newly reserved"})
        local_files = set()
        missing, missing_speakers = [], 0
        for case in cases:
            tokens = {"audio:" + case["sha256"]}
            if case.get("speaker"):
                tokens.add("librispeech:speaker:" + str(case["speaker"]))
                for field in ("source_utt", "ref_utt"):
                    if case.get(field):
                        tokens.add("librispeech:utterance:" + case[field])
            else:
                metadata = mapping.get(case.get("trial_id"), {})
                target = metadata.get("TTS_VC_target_speaker")
                if target and target != "-":
                    tokens.add("vctk:speaker:" + target)
                for field in ("VCTK_ID", "VC_source_VCTK_ID"):
                    utterance = metadata.get(field)
                    if utterance and utterance != "-":
                        tokens.add("vctk:utterance:" + utterance)
                        tokens.add("vctk:speaker:" + utterance.split("_")[0])
                if case.get("source_group"):
                    tokens.add("asvspoof:source:" + case["source_group"])
                if not any(t.startswith("vctk:speaker:") for t in tokens):
                    missing_speakers += 1
            variants = [case, *case.get("channels", {}).values()] if isinstance(case.get("channels"), dict) else [case]
            for variant in variants:
                tokens.add("audio:" + variant["sha256"])
                path = checked_path(root, variant["path"])
                local_files.add(path)
                if not path.is_file():
                    missing.append(str(path.relative_to(root)))
            records.append({"record_id": name + ":" + case["id"], "manifest": name,
                            "split": case["split"], "label": case["label"],
                            "original_sha256": case["sha256"], "identity_tokens": sorted(tokens)})
        summary.update({"referenced_local_files": len(local_files), "missing_files": sorted(set(missing)),
                        "referenced_bytes_present": sum(p.stat().st_size for p in local_files if p.is_file()),
                        "missing_speaker_identity": missing_speakers})
        if missing:
            errors.append(f"Missing audio referenced by {name}")
    groups = connected_groups(records)
    indexed = {r["record_id"]: r for r in records}
    cross_manifest, cross_split = [], []
    for group in groups:
        group_id = sha256("\n".join(group).encode()).hexdigest()[:16]
        sources = sorted({indexed[r]["manifest"] for r in group})
        if len(sources) > 1:
            cross_manifest.append({"group_id": group_id, "manifests": sources, "case_rows": len(group)})
        for source in sources:
            splits = sorted({indexed[r]["split"] for r in group if indexed[r]["manifest"] == source})
            if len(splits) > 1:
                members = [r for r in group if indexed[r]["manifest"] == source]
                cross_split.append({"group_id": group_id, "manifest": source, "splits": splits,
                                    "case_rows": len(members), "record_ids": members})
    material = []
    for relative in ("data/asvspoof/audio", "data/librispeech/LibriSpeech", "data/modern/audio", "data/modern-v4/audio"):
        base = root / relative
        files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3")) if base.exists() else []
        material.append({"path": relative, "audio_files": len(files), "bytes": sum(p.stat().st_size for p in files),
                         "new_locked_test_eligibility": "unassessed; file presence does not establish provenance or independence"})
    by_hash = defaultdict(set)
    for row in records:
        by_hash[row["original_sha256"]].add(row["label"])
    conflicts = sum(len(labels) > 1 for labels in by_hash.values())
    if conflicts:
        errors.append("Same source bytes have conflicting labels")
    method_runs, method_records = {}, []
    for path in sorted((root / "data/evaluation/method-runs").glob("*.json")):
        path = checked_path(root, path)
        raw = path.read_bytes()
        data = json.loads(raw)
        if (sha256(raw.rstrip(b"\n")).hexdigest() != path.stem
                or (data.get("schema_version"), data.get("mode")) not in {
                    ("method-benchmark-result-1.0", "supplied development annotations"),
                    ("core-detector-screen-1.0", "supplied development authenticity labels"),
                    ("core-preprocessing-experiment-1.0", "supplied development authenticity labels")}
                or not data.get("input_identities")):
            raise ValueError(f"Invalid prior-use comparison receipt: {path.name}")
        method_runs[str(path.relative_to(root))] = sha256(raw).hexdigest()
        for record in data["input_identities"]:
            if (not isinstance(record.get("record_id"), str) or not isinstance(record.get("identity_tokens"), list)
                    or not record["identity_tokens"] or any(not isinstance(t, str) or not t for t in record["identity_tokens"])):
                raise ValueError("Invalid prior-use identity record")
            method_records.append({"record_id": "method-run:" + path.stem + ":" + record["record_id"],
                                   "identity_tokens": record["identity_tokens"]})
    result = {
        "schema_version": "corpus-inventory-1.0", "mode": "local metadata, transitive grouping and file existence; no inference or downloads",
        "manifests": manifests, "historical_case_rows": len(records), "unique_original_byte_hashes": len(by_hash),
        "connected_identity_groups": len(groups), "groups_spanning_manifests": cross_manifest,
        "groups_spanning_splits_within_a_manifest": cross_split, "conflicting_byte_labels": conflicts,
        "local_material": material, "eligible_new_locked_groups": 0,
        "errors": errors, "inventory_complete": not errors,
        "mapping_sha256": sha256(mapping_path.read_bytes()).hexdigest() if mapping_path.is_file() else None,
        "method_run_file_sha256": method_runs,
        "limitations": ["Group counts depend on supplied identities; missing speakers and perceptual duplicates remain unresolved.",
                        "Historical split overlap is reported, not silently repaired or used to reassign data.",
                        "Byte identities are taken from manifests; this inventory does not verify audio content hashes.",
                        "No unmanifested local recording is automatically designated an independent or provenance-backed test.",
                        "No claim is made about absence from upstream model training."],
    }
    if include_records:
        result["identity_records"] = records + method_records
    return result
