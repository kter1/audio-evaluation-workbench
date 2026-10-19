"""Blind, portable reviewer packages. No references, scores, or source names."""
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import uuid
import zipfile

from .audio import decode, wav_bytes
from .store import canonical, digest, now


def export_package(store, reviewer, phase, count):
    if not reviewer.strip() or reviewer == store.setting("primary_reviewer"):
        raise ValueError("Use a distinct independent reviewer name")
    if count <= 0:
        raise ValueError("Choose at least one recording")
    if store.one("SELECT 1 FROM assignments WHERE reviewer=? AND phase=? AND role='independent'", (reviewer, phase)):
        raise ValueError("That reviewer already has an independent package for this phase")
    rows = store.rows("SELECT * FROM assignments WHERE role='primary' AND phase=? ORDER BY ordinal", (phase,))
    if any(a["revealed_at"] for a in rows) or count > len(rows):
        raise ValueError("Choose an unrevealed phase with enough recordings")
    rng = random.Random(store.study_manifest().get("seed", 0) + 93)
    chosen = rng.sample(rows, count)
    package_id = uuid.uuid4().hex
    manifest = {"format": "audio-review-lab-blind-v1", "package_id": package_id,
                "reviewer": reviewer, "dataset_kind": store.setting("dataset_kind"), "assignments": [], "guides": {}}
    stream = BytesIO()
    pending = []
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for order, a in enumerate(chosen):
            audio, rate = store.validate_media(a["recording_id"])
            blob = wav_bytes(audio, rate)
            aid = uuid.uuid4().hex
            manifest["assignments"].append({"id": aid, "phase": phase, "guideline": a["guideline"],
                                            "media": f"audio/{aid}.wav", "sha256": hashlib.sha256(blob).hexdigest(), "ordinal": order})
            manifest["guides"][a["guideline"]] = store.one("SELECT text FROM guidelines WHERE version=?", (a["guideline"],))["text"]
            archive.writestr(f"audio/{aid}.wav", blob)
            pending.append((aid, a["recording_id"], reviewer, phase, "independent", a["guideline"], package_id, order))
        archive.writestr("manifest.json", canonical(manifest))
        credit = ("Generated non-speech test tones. All labels and scores are arbitrary software fixtures; not research evidence."
                  if manifest["dataset_kind"] == "demo" else
                  "ASVspoof 2021 Logical Access. Original data creators: ASVspoof challenge consortium; full creator credits at https://zenodo.org/records/4837263 . Data source: https://www.asvspoof.org/index2021.html . ODC-By 1.0: https://opendatacommons.org/licenses/by/1-0/ . These neutral WAV files are PCM conversions for blind review. Source IDs and reference metadata are withheld by the study owner until independent review is completed.")
        archive.writestr("ATTRIBUTION.txt", credit)
    with store.conn:
        store.conn.execute("INSERT INTO packages(id,manifest) VALUES(?,?)", (package_id, canonical(manifest)))
        store.conn.executemany("INSERT INTO assignments(id,recording_id,reviewer,phase,role,guideline,package_id,ordinal) VALUES(?,?,?,?,?,?,?,?)", pending)
    return stream.getvalue()


def load_package(store, data):
    if store.one("SELECT 1 FROM recordings LIMIT 1"):
        raise ValueError("Load a reviewer package into a new database")
    target = store.path.parent / (store.path.stem + "_media")
    with zipfile.ZipFile(BytesIO(data)) as archive:
        if sum(i.file_size for i in archive.infolist()) > 200_000_000:
            raise ValueError("Reviewer package exceeds 200 MB")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format"] != "audio-review-lab-blind-v1" or not manifest["reviewer"].strip():
            raise ValueError("Unsupported reviewer package")
        items = manifest["assignments"]
        if not items or len({a["id"] for a in items}) != len(items):
            raise ValueError("Package contains no assignments or duplicate IDs")
        media = []
        for a in items:
            if len(a["id"]) != 32 or any(c not in "0123456789abcdef" for c in a["id"]) or a["media"] != f"audio/{a['id']}.wav":
                raise ValueError("Invalid package media path")
            if a["guideline"] not in manifest["guides"]:
                raise ValueError("Missing guideline")
            blob = archive.read(a["media"])
            if hashlib.sha256(blob).hexdigest() != a["sha256"]:
                raise ValueError("Reviewer audio checksum mismatch")
            audio, rate = decode(BytesIO(blob))
            media.append((a, blob, len(audio) / rate, rate))
    # All package contents validated before database mutation. Paths are generated locally.
    target.mkdir(parents=True, exist_ok=True)
    with store.conn:
        for version, text in manifest["guides"].items():
            store.conn.execute("INSERT INTO guidelines VALUES(?,?,?)", (version, text, hashlib.sha256(text.encode()).hexdigest()))
        for a, blob, duration, rate in media:
            path = target / (a["id"] + ".wav")
            path.write_bytes(blob)
            store.conn.execute("INSERT INTO recordings(id,trial_id,path,subset,codec,attack,kind,sha256,duration,sample_rate) VALUES(?,?,?,?,?,?,?,?,?,?)", (a["id"], a["id"], str(path), "reviewer", "withheld", "withheld", "reviewer", a["sha256"], duration, rate))
            store.conn.execute("INSERT INTO assignments(id,recording_id,reviewer,phase,role,guideline,package_id,ordinal) VALUES(?,?,?,?,?,?,?,?)", (a["id"], a["id"], manifest["reviewer"], a["phase"], "independent", a["guideline"], manifest["package_id"], a["ordinal"]))
        store.conn.execute("INSERT INTO packages(id,manifest) VALUES(?,?)", (manifest["package_id"], canonical(manifest)))
        store.set_setting("reviewer_mode", True)
        store.set_setting("primary_reviewer", manifest["reviewer"])
        store.set_setting("dataset_kind", manifest["dataset_kind"])
    return manifest


def package_bytes(store, package_id):
    """Rebuild an existing blind package without assigning duplicate work."""
    package = store.one("SELECT manifest FROM packages WHERE id=?", (package_id,))
    if not package or store.setting("reviewer_mode"):
        raise ValueError("Unknown owner-side reviewer package")
    manifest = json.loads(package["manifest"])
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", canonical(manifest))
        archive.writestr("ATTRIBUTION.txt", "ASVspoof 2021 Logical Access: https://zenodo.org/records/4837263 . ODC-By 1.0. Neutral PCM WAV conversions for blind review. Generated demo tones, when identified as demo in manifest.json, are software fixtures only.")
        for item in manifest["assignments"]:
            a = store.assignment(item["id"])
            audio, rate = store.validate_media(a["recording_id"])
            blob = wav_bytes(audio, rate)
            if hashlib.sha256(blob).hexdigest() != item["sha256"]:
                raise ValueError("Packaged audio no longer matches its original checksum")
            archive.writestr(item["media"], blob)
    return stream.getvalue()


def export_results(store):
    if not store.setting("reviewer_mode"):
        raise ValueError("Results are exported from an independent reviewer database")
    package = store.one("SELECT * FROM packages")
    manifest = json.loads(package["manifest"])
    reviews = []
    for item in manifest["assignments"]:
        a = store.assignment(item["id"])
        if not a["submitted_at"] or a["revealed_at"]:
            raise ValueError("Submit all package assignments before exporting")
        r = store.one("SELECT payload FROM annotation_revisions WHERE assignment_id=? AND event='submitted'", (a["id"],))
        reviews.append({"id": a["id"], "submitted_at": a["submitted_at"], "annotation": json.loads(r["payload"])})
    return canonical({"format": "audio-review-lab-results-v1", "package_id": package["id"],
                      "manifest_sha256": digest(manifest), "reviewer": manifest["reviewer"], "reviews": reviews}).encode()


def import_results(store, data):
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Results must be a JSON object")
    package = store.one("SELECT * FROM packages WHERE id=?", (result.get("package_id"),))
    if not package or result.get("format") != "audio-review-lab-results-v1":
        raise ValueError("Unknown reviewer package or results format")
    fingerprint = digest(result)
    if package["results_hash"]:
        if package["results_hash"] == fingerprint:
            return 0  # An identical resend is safe and idempotent.
        raise ValueError("Conflicting results already imported for this package")
    manifest = json.loads(package["manifest"])
    if result["manifest_sha256"] != digest(manifest) or result["reviewer"] != manifest["reviewer"]:
        raise ValueError("Reviewer or package manifest mismatch")
    expected = {a["id"] for a in manifest["assignments"]}
    reviews = result["reviews"]
    if not isinstance(reviews, list) or any(not isinstance(r, dict) or set(r) != {"id", "submitted_at", "annotation"} for r in reviews):
        raise ValueError("Malformed reviewer results")
    if len(reviews) != len(expected) or {r["id"] for r in reviews} != expected:
        raise ValueError("Results must contain each assigned review exactly once")
    for r in reviews:
        a = store.assignment(r["id"])
        if a["submitted_at"] or a["revealed_at"]:
            raise ValueError("Assignment already submitted or revealed")
        if not isinstance(r["submitted_at"], str):
            raise ValueError("Invalid submission timestamp")
        timestamp = datetime.fromisoformat(r["submitted_at"])
        if timestamp.tzinfo is None or timestamp > datetime.fromisoformat(now()):
            raise ValueError("Invalid submission timestamp")
        audio, rate = store.validate_media(a["recording_id"])
        store.validate_annotation(r["annotation"], len(audio) / rate)
    with store.conn:
        for r in reviews:
            store.conn.execute("INSERT INTO annotation_revisions(assignment_id,event,payload,created_at) VALUES(?,?,?,?)", (r["id"], "submitted", canonical(r["annotation"]), r["submitted_at"]))
            store.conn.execute("UPDATE assignments SET submitted_at=? WHERE id=?", (r["submitted_at"], r["id"]))
        store.conn.execute("UPDATE packages SET results_hash=? WHERE id=?", (fingerprint, package["id"]))
    return len(reviews)
