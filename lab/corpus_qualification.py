"""Fail-closed structural qualification of prospective detection corpora.

Does not score audio or verify the truth of an independent reviewer's attestations.
Missing audits, history, identities or sample support prevent qualification.
"""
from collections import defaultdict
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re

import soundfile as sf

from lab.baseline_audit import checked_path
from lab.corpus_inventory import connected_groups

PARTITIONS = {"development", "calibration", "locked_test"}
AUDITS = ("perceptual_duplicates", "label_provenance", "training_exposure", "previous_use")


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def qualify(document, root, contract, history=None):
    root = Path(root).resolve()
    errors, exclusions = [], []
    cases = document.get("cases", [])
    if document.get("schema_version") != "prospective-corpus-1.0" or not cases:
        raise ValueError("Expected a non-empty prospective-corpus-1.0 manifest")
    required = set(contract["partitions"]["manifest_required"]) | {"label", "path", "conditions", "source_references"}
    records, indexed, actual_hashes = [], {}, {}
    for case in cases:
        name = case.get("id")
        if not isinstance(name, str) or not name.strip() or name in indexed:
            raise ValueError("Case IDs must be unique non-empty strings")
        indexed[name] = case
        if not required <= case.keys():
            raise ValueError(f"{name}: missing fields {sorted(required - case.keys())}")
        for key in required - {"parent_id", "source_references", "conditions", "duration_seconds"}:
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError(f"{name}: {key} must be a non-empty string")
        if case["partition"] not in PARTITIONS or case["label"] not in {"human", "synthetic"}:
            raise ValueError(f"{name}: invalid partition/label")
        if (case["label"] == "human") != (case["generator_family"] == "bonafide"):
            raise ValueError(f"{name}: human references require bonafide; synthetic references require a named generator family")
        duration = case["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{name}: invalid duration")
        if not re.fullmatch(r"[0-9a-f]{64}", case["original_sha256"]):
            raise ValueError(f"{name}: invalid original_sha256")
        if case["parent_id"] is not None and (not isinstance(case["parent_id"], str) or not case["parent_id"].strip()):
            raise ValueError(f"{name}: parent_id must be null or a case ID")
        for field in ("conditions", "source_references"):
            if not isinstance(case[field], list) or any(not isinstance(t, str) or not t.strip() for t in case[field]):
                raise ValueError(f"{name}: {field} must be an array of identity/condition strings")
        for identity in (case["source_group"], case["speaker_group"], *case["source_references"]):
            if ":" not in identity:
                raise ValueError(f"{name}: identity values must be dataset-namespaced")
        if case["label_provenance"].casefold() in {"unknown", "unverified", "inferred"}:
            exclusions.append(name)
            errors.append(f"{name}: reference label provenance is unresolved")
        path = checked_path(root, case["path"])
        if not path.is_file():
            errors.append(f"{name}: missing audio")
        else:
            actual = file_hash(path)
            actual_hashes[name] = actual
            if actual != case["original_sha256"]:
                errors.append(f"{name}: source byte hash mismatch")
            try:
                info = sf.info(path)
                if abs(info.duration - duration) > 1 / info.samplerate:
                    errors.append(f"{name}: declared duration differs by more than one source sample")
                if not .5 <= info.duration <= 600 or not 8000 <= info.samplerate <= 192000 or not 1 <= info.channels <= 8 or info.frames * info.channels > 60_000_000:
                    errors.append(f"{name}: audio outside supported analysis limits")
            except (OSError, RuntimeError) as exc:
                errors.append(f"{name}: unreadable audio header ({type(exc).__name__})")
        tokens = {"audio:" + case["original_sha256"], case["speaker_group"], case["source_group"], *case["source_references"]}
        if name in actual_hashes:
            tokens.add("audio:" + actual_hashes[name])
        tokens.add("case:" + name)
        if case["parent_id"] is not None:
            tokens.add("case:" + case["parent_id"])
        records.append({"record_id": name, "identity_tokens": sorted(tokens)})
    for name, case in indexed.items():
        parent = case["parent_id"]
        chain = {name}
        while parent is not None:
            if parent not in indexed:
                errors.append(f"{name}: missing parent {parent}")
                break
            if parent in chain:
                errors.append(f"{name}: cyclic parent relationship")
                break
            chain.add(parent)
            parent = indexed[parent]["parent_id"]
    groups = connected_groups(records)
    group_ids = {}
    for group in groups:
        group_id = sha256("\n".join(group).encode()).hexdigest()[:16]
        for name in group:
            group_ids[name] = group_id
        splits = {indexed[name]["partition"] for name in group}
        if len(splits) > 1:
            errors.append(f"Connected identity group crosses partitions: {group_id}")
    by_audio = defaultdict(set)
    for case in cases:
        by_audio[actual_hashes.get(case["id"], case["original_sha256"])].add(case["label"])
    if any(len(labels) > 1 for labels in by_audio.values()):
        errors.append("Identical source bytes have conflicting reference labels")

    history_matches = []
    if not history or history.get("schema_version") != "prior-use-registry-1.0" or not history.get("records"):
        errors.append("A non-empty prior-use registry is required")
    else:
        old_tokens = {token for row in history["records"] for token in row["identity_tokens"]}
        # Check every partition: historical data must be explicitly separated
        # rather than silently repurposed as a new prospective corpus.
        for row in records:
            common = sorted(set(row["identity_tokens"]) & old_tokens)
            if common:
                history_matches.append({"id": row["record_id"], "identities": common})
        if history_matches:
            errors.append("Prospective corpus overlaps prior-use identities")
    audit_hashes = {}
    for name in AUDITS:
        entry = document.get("audits", {}).get(name, {})
        if entry.get("status") != "passed" or not entry.get("reviewer"):
            errors.append(f"Independent audit unresolved: {name}")
            continue
        if name == "training_exposure" and entry.get("conclusion") != "documented_no_overlap":
            errors.append("Upstream training exposure is unknown or overlapping")
        if not entry.get("path") or not entry.get("sha256"):
            errors.append(f"Audit evidence missing: {name}")
            continue
        path = checked_path(root, entry["path"])
        if not path.is_file() or file_hash(path) != entry["sha256"]:
            errors.append(f"Audit evidence missing or changed: {name}")
        else:
            audit_hashes[name] = entry["sha256"]

    training = [c for c in cases if c["partition"] != "locked_test"]
    locked = [c for c in cases if c["partition"] == "locked_test"]
    if not all(any(c["partition"] == p for c in cases) for p in PARTITIONS):
        errors.append("Development, calibration and locked-test partitions are all required")
    for field, declared in (("generator_family", "heldout_generator_families"), ("domain", "heldout_domains")):
        values = document.get(declared, [])
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            errors.append(f"Declare non-empty {declared}")
            continue
        for value in values:
            if any(c[field] == value for c in training) or not any(c[field] == value for c in locked):
                errors.append(f"Declared holdout is absent from test or present in development/calibration: {field}/{value}")
    bar = contract["synthetic_speech"]
    support = {}
    for condition in bar["critical_conditions"]:
        part = [c for c in locked if condition in c["conditions"] and c["id"] not in exclusions]
        support[condition] = {}
        for label, minimum in (("human", bar["minimum_independent_human_groups_each_critical_condition"]),
                               ("synthetic", bar["minimum_independent_synthetic_groups_each_critical_condition"])):
            count = len({group_ids[c["id"]] for c in part if c["label"] == label})
            support[condition][label] = {"connected_groups": count, "required": minimum}
            if count < minimum:
                errors.append(f"Insufficient independent support: {condition}/{label} {count} < {minimum}")
    return {"schema_version": "corpus-qualification-1.0", "case_rows": len(cases),
            "connected_groups": len(groups), "group_membership": group_ids,
            "history_matches": history_matches, "excluded_unresolved_labels": exclusions,
            "critical_condition_support": support, "verified_audit_file_sha256": audit_hashes,
            "errors": errors, "structural_gates_passed": not errors,
            "scope": "Structural/evidence-file qualification only; reviewer assertions are not independently proven by this program.",
            "limitations": ["Header and byte hashes are checked; full decoder validity is checked by the analysis runner.",
                            "No detection performance, calibration or world-class claim follows from qualification.",
                            "History scope is limited to identities supplied by the prior-use registry."]}


def qualify_file(path, root, history_path=None):
    path, root = Path(path), Path(root)
    contract_path = root / "docs/evaluation-contract-v1.json"
    document = json.loads(path.read_text())
    history = json.loads(Path(history_path).read_text()) if history_path else None
    result = qualify(document, root, json.loads(contract_path.read_text()), history)
    if history:
        from lab.corpus_inventory import MANIFESTS
        for name in MANIFESTS:
            relative = "data/evaluation/" + name
            recorded = history.get("manifest_file_sha256", {}).get(relative)
            local = root / relative
            if not recorded or not local.is_file() or file_hash(local) != recorded:
                result["errors"].append(f"Prior-use registry missing/stale historical manifest evidence: {name}")
        mapping = root / "data/asvspoof/source_mapping.tsv"
        if not mapping.is_file() or file_hash(mapping) != history.get("mapping_sha256"):
            result["errors"].append("Prior-use registry mapping evidence is missing or stale")
        current_runs = {str(p.relative_to(root)): file_hash(checked_path(root, p))
                        for p in sorted((root / "data/evaluation/method-runs").glob("*.json"))}
        if current_runs != history.get("method_run_file_sha256", {}):
            result["errors"].append("Prior-use registry is stale after method comparisons; rebuild it before qualification")
        result["structural_gates_passed"] = not result["errors"]
    result.update(manifest_sha256=file_hash(path), contract_sha256=file_hash(contract_path),
                  history_registry_sha256=file_hash(Path(history_path)) if history_path else None,
                  qualification_code_sha256=file_hash(Path(__file__)))
    return result
