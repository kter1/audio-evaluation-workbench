"""Local-first research dataset QA with provenance and append-only investigations.

This is separate from the legacy blind annotation study. Dataset input is intentionally
immutable; observations, detector hypotheses and adjudications never replace references.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import StringIO
from pathlib import Path
import csv
import json
import math
import re
import sqlite3

RULESET = "1.0"
LABELS = {"human", "synthetic", "uncertain"}
SEVERITIES = {"error", "warning", "review"}
STATUSES = {"open", "investigating", "resolved", "dismissed", "escalated"}
LIMIT_RECORDS = 5000
LIMIT_ANNOTATIONS = 20000
MAX_MANIFEST_BYTES = 5_000_000

RULES = {
    "MISSING_ID": ("error", "Provide a non-empty recording ID."),
    "DUPLICATE_ID": ("error", "Give each manifest row a unique recording ID."),
    "BAD_DURATION": ("error", "Provide a finite recording duration greater than zero."),
    "INVALID_MEDIA_PATH": ("error", "Use a relative path inside the declared media root."),
    "MISSING_MEDIA": ("warning", "Locate or reattach the original recording."),
    "BAD_SHA256": ("error", "Correct the claimed SHA-256 digest."),
    "HASH_MISMATCH": ("error", "Verify original media provenance before continuing."),
    "CODEC_MISMATCH": ("warning", "Check the codec against the actual file format."),
    "MISSING_ANNOTATION_ID": ("error", "Provide a stable annotation ID."),
    "DUPLICATE_ANNOTATION_ID": ("error", "Assign unique IDs to annotation events."),
    "UNKNOWN_RECORDING": ("error", "Correct the annotation's recording reference."),
    "UNKNOWN_LABEL": ("error", "Use a label from the declared human/synthetic/uncertain taxonomy."),
    "BAD_TIMESPAN": ("error", "Correct the timestamp range using the recording duration."),
    "MISSING_SOURCE": ("error", "Identify the source or annotator for the label."),
    "MISSING_GUIDELINE": ("warning", "Record the annotation guideline version."),
    "MISSING_ESCALATION_REASON": ("error", "Explain why the case requires escalation."),
    "CONFLICTING_LABELS": ("review", "Review differing human and synthetic labels; do not assume either is wrong."),
}
SQL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS datasets (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
 origin TEXT NOT NULL, manifest_sha TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recording_rows (
 dataset_id TEXT NOT NULL REFERENCES datasets(id), row_index INTEGER NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(dataset_id,row_index));
CREATE TABLE IF NOT EXISTS annotation_rows (
 dataset_id TEXT NOT NULL REFERENCES datasets(id), row_index INTEGER NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(dataset_id,row_index));
CREATE TABLE IF NOT EXISTS audit_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL REFERENCES datasets(id),
 ruleset TEXT NOT NULL, timestamp TEXT NOT NULL, flag_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS issues (
 id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id),
 rule_id TEXT NOT NULL, severity TEXT NOT NULL, recording_id TEXT,
 annotation_id TEXT, row_index INTEGER, evidence TEXT NOT NULL,
 next_check TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS issues_by_dataset ON issues(dataset_id,severity,rule_id);
CREATE TABLE IF NOT EXISTS issue_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, issue_id TEXT NOT NULL REFERENCES issues(id),
 status TEXT NOT NULL, actor TEXT NOT NULL, note TEXT NOT NULL,
 timestamp TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_by_issue ON issue_events(issue_id,id);
"""

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def safe_csv(rows, fields):
    out = StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        clean = {}
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, (list, dict)):
                value = canonical(value)
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            clean[field] = value
        writer.writerow(clean)
    return ("\ufeff" + out.getvalue()).encode()

def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None

def _str(value):
    return value.strip() if isinstance(value, str) else ""

def _inside_media_root(root, name):
    """Absolute, traversal and symlink escape paths are never followed."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return None
    p = Path(name)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        return None
    base = Path(root).resolve()
    candidate = (base / p).resolve()
    if not candidate.is_relative_to(base):
        return None
    return candidate

def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    for key in ("dataset_id", "name", "origin", "kind"):
        if not _str(manifest.get(key)):
            raise ValueError("Manifest is missing " + key)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", manifest["dataset_id"]):
        raise ValueError("dataset_id must be a short safe identifier.")
    records, annotations = manifest.get("recordings"), manifest.get("annotations")
    if not isinstance(records, list) or not isinstance(annotations, list):
        raise ValueError("Manifest must contain recordings and annotations arrays.")
    if not records or len(records) > LIMIT_RECORDS or len(annotations) > LIMIT_ANNOTATIONS:
        raise ValueError("Dataset exceeds allowed size or is empty.")
    if any(not isinstance(row, dict) for row in records + annotations):
        raise ValueError("Every recording and annotation must be an object.")
    try:
        payload = canonical(manifest)
    except (ValueError, TypeError) as exc:
        raise ValueError("Manifest contains non-finite or invalid data.") from exc
    if len(payload.encode()) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds 5 MB.")
    return sha256(payload.encode()).hexdigest()

class ResearchDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SQL)
        self.conn.execute("PRAGMA busy_timeout=30000")

    def close(self):
        self.conn.close()

    def rows(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args)]

    def one(self, sql, args=()):
        r = self.conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    def datasets(self):
        return self.rows("SELECT * FROM datasets ORDER BY created_at DESC,id")

    def import_manifest(self, manifest):
        digest = validate_manifest(manifest)
        dataset_id = manifest["dataset_id"]
        old = self.one("SELECT manifest_sha FROM datasets WHERE id=?", (dataset_id,))
        if old:
            if old["manifest_sha"] != digest:
                raise ValueError("Dataset ID already exists with different content. Use a new versioned ID.")
            return False
        with self.conn:
            self.conn.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?)",
                (dataset_id, manifest["name"], manifest["kind"], manifest["origin"], digest, timestamp()))
            self.conn.executemany("INSERT INTO recording_rows VALUES(?,?,?)",
                [(dataset_id, i, canonical(r)) for i, r in enumerate(manifest["recordings"])])
            self.conn.executemany("INSERT INTO annotation_rows VALUES(?,?,?)",
                [(dataset_id, i, canonical(r)) for i, r in enumerate(manifest["annotations"])])
        return True

    def manifest(self, dataset_id):
        d = self.one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
        if not d:
            raise ValueError("Unknown dataset")
        records = [json.loads(r["payload"]) for r in self.rows(
            "SELECT payload FROM recording_rows WHERE dataset_id=? ORDER BY row_index", (dataset_id,))]
        annotations = [json.loads(r["payload"]) for r in self.rows(
            "SELECT payload FROM annotation_rows WHERE dataset_id=? ORDER BY row_index", (dataset_id,))]
        return d, records, annotations

    def audit(self, dataset_id, media_root=None):
        d, records, annotations = self.manifest(dataset_id)
        flags = {}
        def flag(rule, record="", annotation="", row_index=None, evidence=None):
            severity, next_check = RULES[rule]
            evidence = evidence or {}
            identity = canonical([dataset_id, rule, record, annotation, row_index, evidence])
            issue_id = sha256(identity.encode()).hexdigest()[:24]
            flags[issue_id] = (issue_id, dataset_id, rule, severity, record or None,
                               annotation or None, row_index, canonical(evidence), next_check, timestamp())

        first = {}
        lengths = {}
        for i, r in enumerate(records):
            rid = _str(r.get("recording_id"))
            if not rid:
                flag("MISSING_ID", row_index=i, evidence={"row": i + 1})
                continue
            if rid in first:
                flag("DUPLICATE_ID", rid, row_index=i,
                     evidence={"first_row": first[rid] + 1, "duplicate_row": i + 1})
            else:
                first[rid] = i
            duration = _number(r.get("duration_seconds"))
            if duration is None or duration <= 0:
                flag("BAD_DURATION", rid, row_index=i, evidence={"value": str(r.get("duration_seconds"))[:100]})
            else:
                lengths[rid] = duration
            relative = r.get("media_path")
            if relative not in (None, ""):
                if not isinstance(relative, str) or _inside_media_root(media_root or ".", relative) is None:
                    flag("INVALID_MEDIA_PATH", rid, row_index=i, evidence={"row": i + 1})
                else:
                    path = _inside_media_root(media_root or ".", relative)
                    if media_root is not None:
                        if not path.is_file():
                            flag("MISSING_MEDIA", rid, row_index=i, evidence={"media_path": relative})
                        else:
                            if path.stat().st_size > 50 * 1024 * 1024:
                                flag("MISSING_MEDIA", rid, row_index=i,
                                     evidence={"reason": "File exceeds 50 MB"})
                            else:
                                declared = _str(r.get("sha256"))
                                if declared and re.fullmatch(r"[0-9a-fA-F]{64}", declared):
                                    actual = sha256(path.read_bytes()).hexdigest()
                                    if actual.lower() != declared.lower():
                                        flag("HASH_MISMATCH", rid, row_index=i,
                                            evidence={"expected": declared, "actual": actual})
                    declared_codec = _str(r.get("codec")).lower()
                    suffix = Path(relative).suffix.lower().lstrip(".")
                    if declared_codec in {"wav", "flac", "mp3", "ogg", "aiff"} and suffix in {"wav", "flac", "mp3", "ogg", "aiff"} and declared_codec != suffix:
                        flag("CODEC_MISMATCH", rid, row_index=i,
                             evidence={"declared_codec": declared_codec, "file_suffix": suffix})
            checksum = r.get("sha256")
            if checksum not in (None, "") and (not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum)):
                flag("BAD_SHA256", rid, row_index=i, evidence={"row": i + 1})

        seen_ann = {}
        by_record = defaultdict(lambda: defaultdict(list))
        for i, a in enumerate(annotations):
            aid = _str(a.get("annotation_id"))
            rid = _str(a.get("recording_id"))
            if not aid:
                flag("MISSING_ANNOTATION_ID", rid, row_index=i, evidence={"annotation_row": i + 1})
            elif aid in seen_ann:
                flag("DUPLICATE_ANNOTATION_ID", rid, aid, i,
                     {"first_row": seen_ann[aid] + 1, "duplicate_row": i + 1})
            else:
                seen_ann[aid] = i
            if rid not in first:
                flag("UNKNOWN_RECORDING", rid, aid, i, {"annotation_row": i + 1})
            label = _str(a.get("label"))
            if label not in LABELS:
                flag("UNKNOWN_LABEL", rid, aid, i, {"label": label[:100]})
            source = _str(a.get("source"))
            if not source:
                flag("MISSING_SOURCE", rid, aid, i, {"annotation_row": i + 1})
            if not _str(a.get("guideline_version")):
                flag("MISSING_GUIDELINE", rid, aid, i, {"annotation_row": i + 1})
            start, end = a.get("start_seconds"), a.get("end_seconds")
            if start is not None or end is not None:
                s, e = _number(start), _number(end)
                duration = lengths.get(rid)
                if s is None or e is None or s < 0 or e <= s or (duration is not None and e > duration):
                    flag("BAD_TIMESPAN", rid, aid, i,
                         {"start": str(start)[:80], "end": str(end)[:80], "duration": duration})
            if a.get("escalated") is True and not _str(a.get("escalation_reason")):
                flag("MISSING_ESCALATION_REASON", rid, aid, i, {"annotation_row": i + 1})
            if rid in first and label in {"human", "synthetic"} and source:
                by_record[rid][label].append({"annotation_id": aid, "source": source})
        for rid, votes in by_record.items():
            if votes["human"] and votes["synthetic"]:
                flag("CONFLICTING_LABELS", rid, evidence={
                    "human_annotations": votes["human"], "synthetic_annotations": votes["synthetic"]})
        with self.conn:
            for item in flags.values():
                self.conn.execute("""INSERT OR IGNORE INTO issues
                    (id,dataset_id,rule_id,severity,recording_id,annotation_id,row_index,evidence,next_check,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""", item)
            self.conn.execute("INSERT INTO audit_runs(dataset_id,ruleset,timestamp,flag_count) VALUES(?,?,?,?)",
                              (dataset_id, RULESET, timestamp(), len(flags)))
        return len(flags)

    def issues(self, dataset_id, severity=None, status=None):
        if not self.one("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)):
            raise ValueError("Unknown dataset")
        rows = self.rows("""SELECT i.*,COALESCE(
            (SELECT e.status FROM issue_events e WHERE e.issue_id=i.id ORDER BY e.id DESC LIMIT 1),'open'
        ) status FROM issues i WHERE i.dataset_id=? ORDER BY
        CASE i.severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,i.rule_id,i.id""", (dataset_id,))
        for r in rows:
            r["evidence"] = json.loads(r["evidence"])
        if severity:
            rows = [r for r in rows if r["severity"] == severity]
        if status:
            rows = [r for r in rows if r["status"] == status]
        return rows

    def issue(self, issue_id):
        r = self.one("SELECT * FROM issues WHERE id=?", (issue_id,))
        if not r:
            raise ValueError("Unknown issue")
        r["evidence"] = json.loads(r["evidence"])
        r["events"] = self.rows("SELECT status,actor,note,timestamp FROM issue_events WHERE issue_id=? ORDER BY id", (issue_id,))
        r["status"] = r["events"][-1]["status"] if r["events"] else "open"
        return r

    def disposition(self, issue_id, status, actor, note):
        self.issue(issue_id)
        if status not in STATUSES or not _str(actor) or not _str(note):
            raise ValueError("Status, actor and explanation are required.")
        with self.conn:
            self.conn.execute("INSERT INTO issue_events(issue_id,status,actor,note,timestamp) VALUES(?,?,?,?,?)",
                              (issue_id, status, actor.strip(), note.strip(), timestamp()))

    def audit_runs(self, dataset_id):
        return self.rows("SELECT * FROM audit_runs WHERE dataset_id=? ORDER BY id DESC", (dataset_id,))

    def export_bundle(self, dataset_id):
        """CSV/JSON/HTML include evidence and dispositions, not private absolute paths."""
        from io import BytesIO
        from zipfile import ZipFile, ZIP_DEFLATED
        d, records, annotations = self.manifest(dataset_id)
        issues = self.issues(dataset_id)
        events = []
        for r in issues:
            for e in self.issue(r["id"])["events"]:
                events.append({"issue_id": r["id"], **e})
        runs = self.audit_runs(dataset_id)
        summary = {
            "dataset_id": dataset_id, "name": d["name"], "kind": d["kind"], "origin": d["origin"],
            "manifest_sha256": d["manifest_sha"], "ruleset": RULESET,
            "recordings": len(records), "annotation_rows": len(annotations),
            "issues": len(issues), "by_severity": dict(Counter(r["severity"] for r in issues)),
            "by_rule": dict(Counter(r["rule_id"] for r in issues)),
            "by_status": dict(Counter(self.issue(r["id"])["status"] for r in issues)),
            "audit_runs": runs,
            "limitations": "Flags are rule-based data-quality findings, not fraud or audio-authenticity determinations. Demo records are intentionally constructed fixtures. Model benchmark reports are separate historical experiments."
        }
        fields = ["id", "rule_id", "severity", "recording_id", "annotation_id", "row_index",
                  "status", "evidence", "next_check"]
        efields = ["issue_id", "status", "actor", "note", "timestamp"]
        html_rows = "".join("<tr>" + "".join("<td>" + escape(str(r.get(k, ""))) + "</td>"
                            for k in ("rule_id", "severity", "recording_id", "status", "next_check"))
                            + "</tr>" for r in issues)
        html = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Research audit</title>"
                "<style>body{font:16px system-ui;max-width:1050px;margin:3rem auto;padding:0 1rem}"
                "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.5rem}"
                "th{text-align:left;background:#eee}</style><h1>Interaction Research Lab</h1><h2>"
                + escape(d["name"]) + "</h2><p>Dataset: " + escape(dataset_id)
                + " · Origin: " + escape(d["origin"]) + " · Ruleset " + RULESET
                + "</p><p>Recordings: " + str(len(records)) + " · Annotations: "
                + str(len(annotations)) + " · Flags: " + str(len(issues)) + "</p><table><thead>"
                "<tr><th>Rule</th><th>Severity</th><th>Recording</th><th>Status</th><th>Next check</th></tr>"
                "</thead><tbody>" + html_rows + "</tbody></table><p>"
                + escape(summary["limitations"]) + "</p></html>")
        buf = BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as z:
            z.writestr("summary.json", json.dumps(summary, indent=2))
            z.writestr("issues.csv", safe_csv(issues, fields))
            z.writestr("disposition_history.csv", safe_csv(events, efields))
            z.writestr("audit.html", html)
            z.writestr("README.txt", "Dataset QA audit. No original audio or private media paths are exported.\n")
        return buf.getvalue()
