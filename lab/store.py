"""SQLite state machine. Blind projections are explicit allowlists."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import uuid

from .audio import checksum, decode
from .metrics import equal_error_threshold

PHASES = ("calibration", "main-01", "main-02", "validation", "challenge")
JUDGMENTS = ("human", "synthetic", "uncertain")
INTELLIGIBILITY = ("clear", "partly intelligible", "unintelligible", "no speech")
TAGS = ("background noise", "compression artifacts", "reverb or echo", "music or hold audio", "clipping", "dropouts",
        "distortion", "overlap", "other")
# What you heard that leans one way. Recorded before reveal so cue reliability can be measured afterwards.
SYNTHETIC_CUES = ("unnatural prosody or rhythm", "flat or uniform emotion", "metallic or robotic timbre",
                  "missing breaths or mouth sounds", "odd pronunciation or stress", "abrupt splice or transition",
                  "too-clean studio sound")
HUMAN_CUES = ("natural breaths or mouth sounds", "natural hesitation or disfluency", "consistent room acoustics")
CUES = SYNTHETIC_CUES + HUMAN_CUES + ("other cue",)


GUIDE = Path(__file__).resolve().parent.parent / "docs" / "annotation-guide.md"


def guide_version(text):
    """The guide states its own version on its first line, e.g. '# Speech annotation guide · version 1.1'."""
    match = re.search(r"version\s+([0-9][\w.]*)", text.splitlines()[0] if text.strip() else "")
    if not match:
        raise ValueError("The annotation guide's first line must state its version")
    return match.group(1)


def register_current_guide(store):
    """Register docs/annotation-guide.md under its own version and make it the default for new assignments."""
    text = GUIDE.read_text()
    version = guide_version(text)
    store.add_guideline(version, text)
    with store.conn:
        store.set_setting("initial_guideline", version)
    return version


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS provenance(id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS guidelines(version TEXT PRIMARY KEY, text TEXT NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recordings(
 id TEXT PRIMARY KEY, trial_id TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
 source_group TEXT, subset TEXT NOT NULL, codec TEXT NOT NULL, attack TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('demo','research','reviewer')), sha256 TEXT,
 duration REAL, sample_rate INTEGER, media_error TEXT);
CREATE TABLE IF NOT EXISTS reference_labels(recording_id TEXT PRIMARY KEY REFERENCES recordings(id), label TEXT NOT NULL CHECK(label IN ('human','synthetic')));
CREATE TABLE IF NOT EXISTS detector_scores(recording_id TEXT PRIMARY KEY REFERENCES recordings(id), score REAL NOT NULL, provenance_id TEXT NOT NULL REFERENCES provenance(id));
CREATE TABLE IF NOT EXISTS threshold_config(id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS study(id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assignments(
 id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(id),
 reviewer TEXT NOT NULL, phase TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('primary','repeat','independent')),
 guideline TEXT NOT NULL REFERENCES guidelines(version), due_at TEXT,
 origin_id TEXT REFERENCES assignments(id), submitted_at TEXT, revealed_at TEXT,
 package_id TEXT, ordinal INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS annotation_revisions(
 id INTEGER PRIMARY KEY AUTOINCREMENT, assignment_id TEXT NOT NULL REFERENCES assignments(id),
 event TEXT NOT NULL CHECK(event IN ('draft','submitted','correction')), payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS one_submission ON annotation_revisions(assignment_id) WHERE event='submitted';
CREATE TABLE IF NOT EXISTS adjudications(
 id INTEGER PRIMARY KEY AUTOINCREMENT, recording_id TEXT NOT NULL REFERENCES recordings(id),
 reviewer TEXT NOT NULL, decision TEXT NOT NULL, rationale TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY AUTOINCREMENT,
 recording_id TEXT NOT NULL REFERENCES recordings(id), title TEXT NOT NULL,
 observation TEXT NOT NULL, hypothesis TEXT NOT NULL, follow_up TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS packages(id TEXT PRIMARY KEY, manifest TEXT NOT NULL, results_hash TEXT);
CREATE TRIGGER IF NOT EXISTS no_assignment_after_reveal BEFORE INSERT ON assignments
WHEN EXISTS(SELECT 1 FROM assignments WHERE phase=NEW.phase AND revealed_at IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'Cannot assign blind work after phase reveal'); END;
CREATE TRIGGER IF NOT EXISTS no_incomplete_reveal BEFORE UPDATE OF revealed_at ON assignments
WHEN NEW.revealed_at IS NOT NULL AND EXISTS(SELECT 1 FROM assignments WHERE phase=NEW.phase AND submitted_at IS NULL)
BEGIN SELECT RAISE(ABORT, 'Incomplete assigned reviews block reveal'); END;
CREATE TRIGGER IF NOT EXISTS freeze_original BEFORE INSERT ON annotation_revisions
WHEN NEW.event IN ('draft','submitted') AND EXISTS(SELECT 1 FROM assignments WHERE id=NEW.assignment_id AND submitted_at IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'Original submission is frozen'); END;
"""


class Store:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")

    def close(self):
        self.conn.close()

    def rows(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args)]

    def one(self, sql, args=()):
        row = self.conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key, value):
        self.conn.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, canonical(value)))

    def add_guideline(self, version, text, phase=None):
        if not version.strip() or not text.strip():
            raise ValueError("Guideline version and text are required")
        with self.conn:
            existing = self.one("SELECT text FROM guidelines WHERE version=?", (version,))
            if existing and existing["text"] != text:
                raise ValueError("A guideline version is immutable. Register a new version.")
            if phase:
                if phase not in PHASES:
                    raise ValueError("Unknown phase")
                if self.one("SELECT 1 FROM annotation_revisions e JOIN assignments a ON a.id=e.assignment_id WHERE a.phase=?", (phase,)):
                    raise ValueError("Cannot change a phase guideline after reviews have started")
            self.conn.execute("INSERT OR IGNORE INTO guidelines VALUES(?,?,?)", (version, text, hashlib.sha256(text.encode()).hexdigest()))
            if phase:
                self.conn.execute("UPDATE assignments SET guideline=? WHERE phase=?", (version, phase))

    def register(self, records, scores, provenance):
        """Atomic import, including references. Duplicate/unknown IDs are errors."""
        ids = [r["trial_id"] for r in records]
        if len(ids) != len(set(ids)) or len(scores) != len({r["trial_id"] for r in scores}):
            raise ValueError("Duplicate recording or score IDs")
        if set(r["trial_id"] for r in scores) - set(ids):
            raise ValueError("Score IDs do not match the recording manifest")
        import math
        if any(not math.isfinite(r["score"]) for r in scores):
            raise ValueError("Scores must be finite")
        if self.one("SELECT 1 FROM recordings LIMIT 1"):
            raise ValueError("This database already contains recordings; use a new database")
        pid = digest(provenance)
        with self.conn:
            self.conn.execute("INSERT INTO provenance VALUES(?,?,?,?)", (pid, "dataset_import", canonical(provenance), now()))
            keys = {}
            for r in records:
                rid = uuid.uuid4().hex
                keys[r["trial_id"]] = rid
                self.conn.execute("INSERT INTO recordings(id,trial_id,path,source_group,subset,codec,attack,kind) VALUES(?,?,?,?,?,?,?,?)",
                                  (rid, r["trial_id"], str(Path(r["path"]).resolve()), r.get("source_group"), r["subset"], r["codec"], r["attack"], r["kind"]))
                self.conn.execute("INSERT INTO reference_labels VALUES(?,?)", (rid, r["reference"]))
            self.conn.executemany("INSERT INTO detector_scores VALUES(?,?,?)", [(keys[s["trial_id"]], s["score"], pid) for s in scores])
            self.set_setting("dataset_kind", records[0]["kind"] if records else "research")
            self.set_setting("import_provenance", pid)

    def freeze_threshold(self):
        existing = self.one("SELECT config FROM threshold_config WHERE id=1")
        if existing:
            return json.loads(existing["config"])
        if self.one("SELECT 1 FROM study"):
            raise ValueError("Threshold must be frozen before sampling")
        rows = self.rows("SELECT r.trial_id,s.score,l.label AS reference FROM recordings r JOIN reference_labels l ON r.id=l.recording_id JOIN detector_scores s ON r.id=s.recording_id WHERE r.subset='progress' ORDER BY r.trial_id")
        progress = self.one("SELECT COUNT(*) AS n FROM recordings WHERE subset='progress'")["n"]
        if len(rows) != progress or self.one("SELECT 1 FROM recordings WHERE subset='progress' AND source_group IS NULL"):
            raise ValueError("All progress recordings need both scores and source mappings")
        config = equal_error_threshold(rows)
        config.update({"calibration_subset": "progress", "trial_scores_sha256": digest(rows),
                       "baseline": "LFCC-GMM", "import_provenance": self.setting("import_provenance"), "frozen_at": now()})
        with self.conn:
            self.conn.execute("INSERT INTO threshold_config VALUES(1,?)", (canonical(config),))
        return config

    def validate_media(self, rid):
        row = self.one("SELECT * FROM recordings WHERE id=?", (rid,))
        try:
            audio, rate = decode(row["path"])
            sha = checksum(row["path"])
            if row["sha256"] and sha != row["sha256"]:
                raise ValueError("Recording checksum changed since assignment")
        except (OSError, RuntimeError, ValueError):
            with self.conn:
                self.conn.execute("UPDATE recordings SET media_error='Recording unavailable, corrupt, or changed' WHERE id=?", (rid,))
            raise ValueError("Recording unavailable, corrupt, or changed; repair it before submission") from None
        with self.conn:
            self.conn.execute("UPDATE recordings SET duration=?,sample_rate=?,sha256=?,media_error=NULL WHERE id=?", (len(audio) / rate, rate, sha, rid))
        return audio, rate

    def prepare_study(self, reviewer="primary", seed=20261019, counts=(20, 80, 100, 100)):
        if self.one("SELECT 1 FROM study"):
            raise ValueError("Study manifest is frozen; use a new database for a different sample")
        if len(counts) != 4 or any(n <= 0 or n % 2 for n in counts):
            raise ValueError("Four positive even phase sizes are required")
        if not reviewer.strip():
            raise ValueError("Reviewer name required")
        guideline = self.setting("initial_guideline", "1.0")
        if not self.one("SELECT 1 FROM guidelines WHERE version=?", (guideline,)):
            raise ValueError(f"Register guideline {guideline} first")
        self.freeze_threshold()
        progress_groups = {r["source_group"] for r in self.rows("SELECT DISTINCT source_group FROM recordings WHERE subset='progress' AND source_group IS NOT NULL")}
        candidates = self.rows("SELECT r.*,l.label FROM recordings r JOIN reference_labels l ON r.id=l.recording_id JOIN detector_scores s ON r.id=s.recording_id WHERE subset='eval' ORDER BY trial_id")
        candidates = [r for r in candidates if r["source_group"] and r["source_group"] not in progress_groups]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        used, pools, failed = set(), {"human": [], "synthetic": []}, []
        required = sum(counts) // 2
        for r in candidates:
            if r["source_group"] in used or len(pools[r["label"]]) >= required:
                continue
            try:
                self.validate_media(r["id"])
            except ValueError:
                failed.append(r["trial_id"])
                continue
            used.add(r["source_group"])
            pools[r["label"]].append(r)
            if all(len(p) == required for p in pools.values()):
                break
        if any(len(p) < required for p in pools.values()):
            raise ValueError(f"Need {required} readable, mapped, distinct-source recordings per class after excluding progress overlap; found human={len(pools['human'])}, synthetic={len(pools['synthetic'])}. No study created.")
        selected, offset = [], 0
        for phase, count in zip(PHASES, counts):
            batch = pools["human"][offset:offset + count // 2] + pools["synthetic"][offset:offset + count // 2]
            rng.shuffle(batch)
            for order, r in enumerate(batch):
                sha = self.one("SELECT sha256 FROM recordings WHERE id=?", (r["id"],))["sha256"]
                selected.append({"trial_id": r["trial_id"], "source_group": r["source_group"], "phase": phase, "sha256": sha, "ordinal": order})
            offset += count // 2
        manifest = {"seed": seed, "counts": dict(zip(PHASES, counts)), "selected": selected,
                    "excluded_unreadable": failed, "progress_group_count": len(progress_groups),
                    "import_provenance": self.setting("import_provenance"), "reviewer": reviewer}
        by_trial = {r["trial_id"]: r["id"] for r in candidates}
        with self.conn:
            for r in selected:
                self.conn.execute("INSERT INTO assignments(id,recording_id,reviewer,phase,role,guideline,ordinal) VALUES(?,?,?,?,?,?,?)", (uuid.uuid4().hex, by_trial[r["trial_id"]], reviewer, r["phase"], "primary", guideline, r["ordinal"]))
            self.conn.execute("INSERT INTO study VALUES(1,?,?)", (canonical(manifest), digest(manifest)))
            self.set_setting("primary_reviewer", reviewer)
        return manifest

    def queue(self, reviewer):
        # Never add r.* or reference joins to this projection.
        return self.rows("""SELECT a.id,a.phase,a.role,a.guideline,a.due_at,a.submitted_at,a.revealed_at,
             r.media_error, a.ordinal FROM assignments a JOIN recordings r ON r.id=a.recording_id
             WHERE reviewer=? ORDER BY CASE phase WHEN 'calibration' THEN 0 WHEN 'main-01' THEN 1 WHEN 'main-02' THEN 2 ELSE 3 END, a.ordinal,a.role""", (reviewer,))

    def assignment(self, aid):
        row = self.one("SELECT * FROM assignments WHERE id=?", (aid,))
        if not row:
            raise ValueError("Unknown assignment")
        return row

    def blind_review(self, aid):
        a = self.assignment(aid)
        result = {k: a[k] for k in ("id", "reviewer", "phase", "role", "guideline", "due_at", "submitted_at")}
        # Repeat assignments intentionally have no relationship to the first answer here.
        event = None if a["submitted_at"] else self.one("SELECT payload FROM annotation_revisions WHERE assignment_id=? AND event='draft' ORDER BY id DESC LIMIT 1", (aid,))
        result["annotation"] = json.loads(event["payload"]) if event else None
        result["guide"] = self.one("SELECT text FROM guidelines WHERE version=?", (a["guideline"],))["text"]
        return result

    def review_audio(self, aid):
        a = self.assignment(aid)
        if a["due_at"] and now() < a["due_at"]:
            raise ValueError("Repeat review opens 48 hours after the original submission")
        return self.validate_media(a["recording_id"])

    def validate_annotation(self, payload, duration, complete=True):
        import math
        allowed = {"intelligibility", "tags", "cues", "judgment", "confidence", "observations", "rationale", "escalated", "escalation_reason"}
        if not isinstance(payload, dict) or set(payload) != allowed:
            raise ValueError("Annotation fields do not match this guideline")
        if payload["intelligibility"] not in INTELLIGIBILITY or payload["judgment"] not in JUDGMENTS:
            raise ValueError("Invalid classification")
        if type(payload["confidence"]) is not int or not 1 <= payload["confidence"] <= 5:
            raise ValueError("Confidence must be an integer from 1 to 5")
        if not isinstance(payload["tags"], list) or any(tag not in TAGS for tag in payload["tags"]):
            raise ValueError("Invalid interference tags")
        if not isinstance(payload["cues"], list) or any(cue not in CUES for cue in payload["cues"]) or len(set(payload["cues"])) != len(payload["cues"]):
            raise ValueError("Invalid perceptual cues")
        if complete and "other cue" in payload["cues"] and not payload["rationale"].strip():
            raise ValueError("Describe the other cue in your decision notes")
        if type(payload["escalated"]) is not bool:
            raise ValueError("Escalation must be true or false")
        for key in ("rationale", "escalation_reason"):
            if not isinstance(payload[key], str) or len(payload[key]) > 10000:
                raise ValueError("Notes must be text of at most 10,000 characters")
        if complete and payload["judgment"] == "uncertain" and not payload["rationale"].strip():
            raise ValueError("Explain why the classification is uncertain")
        if complete and payload["escalated"] and not payload["escalation_reason"].strip():
            raise ValueError("An escalation reason is required")
        if not isinstance(payload["observations"], list):
            raise ValueError("Observations must be a list")
        for o in payload["observations"]:
            if not isinstance(o, dict) or set(o) != {"start", "end", "note"} or not isinstance(o["note"], str) or not o["note"].strip():
                raise ValueError("Each observation needs start, end, and a note")
            if any(type(o[k]) not in (int, float) or not math.isfinite(o[k]) for k in ("start", "end")) or not 0 <= o["start"] < o["end"] <= duration:
                raise ValueError("Observation timestamps must be finite and inside the recording")
        canonical(payload)

    def save(self, aid, payload, event="draft"):
        if event not in ("draft", "submitted", "correction"):
            raise ValueError("Invalid annotation event")
        a = self.assignment(aid)
        audio, rate = self.review_audio(aid)
        self.validate_annotation(payload, len(audio) / rate, complete=event != "draft")
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            a = self.assignment(aid)
            if a["submitted_at"] and event != "correction":
                raise ValueError("Submission is frozen; record a correction in Quality Review")
            if event == "correction" and not a["revealed_at"]:
                raise ValueError("Corrections require a revealed, frozen submission")
            timestamp = now()
            self.conn.execute("INSERT INTO annotation_revisions(assignment_id,event,payload,created_at) VALUES(?,?,?,?)", (aid, event, canonical(payload), timestamp))
            if event == "submitted":
                self.conn.execute("UPDATE assignments SET submitted_at=? WHERE id=?", (timestamp, aid))

    def schedule_repeats(self, phase="main-01", count=20):
        if self.one("SELECT 1 FROM assignments WHERE role='repeat' AND phase=?", (phase,)):
            raise ValueError("Repeats already scheduled for this phase")
        rows = self.rows("SELECT * FROM assignments WHERE phase=? AND role='primary' ORDER BY ordinal", (phase,))
        if not rows or any(not r["submitted_at"] or r["revealed_at"] for r in rows):
            raise ValueError("Submit the entire phase and keep it blind before scheduling repeats")
        if count <= 0 or count > len(rows):
            raise ValueError("Repeat count must fit the completed phase")
        rng = random.Random(self.study_manifest()["seed"] + 48)
        chosen = rng.sample(rows, count)
        with self.conn:
            for order, r in enumerate(chosen):
                due = (datetime.fromisoformat(r["submitted_at"]) + timedelta(hours=48)).isoformat()
                self.conn.execute("INSERT INTO assignments(id,recording_id,reviewer,phase,role,guideline,due_at,origin_id,ordinal) VALUES(?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, r["recording_id"], r["reviewer"], phase, "repeat", r["guideline"], due, r["id"], order + 10000))
        return len(chosen)

    def reveal_phase(self, phase):
        rows = self.rows("SELECT * FROM assignments WHERE phase=?", (phase,))
        if not rows or any(not r["submitted_at"] for r in rows):
            raise ValueError("All assigned reviews, repeats, and independent reviews must be submitted first")
        # Reserve the first main batch for the planned blind consistency study.
        if phase == "main-01" and not any(r["role"] == "repeat" for r in rows):
            raise ValueError("Schedule and complete blind repeats before revealing main-01")
        with self.conn:
            self.conn.execute("UPDATE assignments SET revealed_at=COALESCE(revealed_at,?) WHERE phase=?", (now(), phase))

    def study_manifest(self):
        row = self.one("SELECT manifest FROM study WHERE id=1")
        return json.loads(row["manifest"]) if row else {}

    def revealed_rows(self):
        return self.rows("""SELECT a.id AS assignment_id,a.recording_id,a.reviewer,a.phase,a.role,a.guideline,
          a.submitted_at,r.trial_id,r.source_group,r.codec,r.attack,r.kind,l.label AS reference,
          s.score,e.payload FROM assignments a JOIN recordings r ON a.recording_id=r.id
          JOIN annotation_revisions e ON e.assignment_id=a.id AND e.event='submitted'
          JOIN reference_labels l ON l.recording_id=r.id
          LEFT JOIN detector_scores s ON s.recording_id=r.id
          WHERE a.revealed_at IS NOT NULL ORDER BY a.phase,a.ordinal,a.role""")

    def correction(self, aid, payload):
        self.save(aid, payload, "correction")

    def adjudicate(self, rid, reviewer, decision, rationale):
        if decision not in JUDGMENTS or not rationale.strip() or not reviewer.strip():
            raise ValueError("Adjudication requires reviewer, decision, and rationale")
        if not self.one("SELECT 1 FROM assignments WHERE recording_id=? AND revealed_at IS NOT NULL", (rid,)):
            raise ValueError("Only revealed cases can be adjudicated")
        with self.conn:
            self.conn.execute("INSERT INTO adjudications(recording_id,reviewer,decision,rationale,created_at) VALUES(?,?,?,?,?)", (rid, reviewer, decision, rationale, now()))

    def add_case(self, rid, title, observation, hypothesis, follow_up):
        if not all(isinstance(s, str) and s.strip() for s in (title, observation, hypothesis, follow_up)):
            raise ValueError("Every case field is required; use 'not established' for an unsupported cause")
        if not self.one("SELECT 1 FROM assignments WHERE recording_id=? AND revealed_at IS NOT NULL", (rid,)):
            raise ValueError("Case studies require a revealed recording")
        with self.conn:
            self.conn.execute("INSERT INTO cases(recording_id,title,observation,hypothesis,follow_up,created_at) VALUES(?,?,?,?,?,?)", (rid, title, observation, hypothesis, follow_up, now()))
