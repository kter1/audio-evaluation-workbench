"""Safe local startup sample catalog, with truthful source descriptions."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import random
import sqlite3
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("fixture", "existing_catalog")


def _generated(index):
    rate = 16000
    duration = (16, 23, 31, 42)[index]
    samples = np.zeros(duration * rate, np.float32)
    time = np.arange(len(samples), dtype=np.float32) / rate
    layouts = [
        ((.5, 5.2, 220, .24), (5.2, 9.5, 880, .51), (11, 15.0, 440, 1.9)),
        ((.6, 9, 180, .3), (10, 13.5, 1300, .2), (16.0, 22, 310, 2.0)),
        ((2, 9, 300, .31), (9.4, 19, 700, .32), (23, 29, 500, 2.2)),
        ((.5, 13, 160, .25), (17, 30, 920, .42), (34, 41, 600, 2.4)),
    ]
    for start, end, hz, gain in layouts[index]:
        a, b = int(start*rate), int(end*rate)
        samples[a:b] = np.clip(gain * np.sin(2*np.pi*hz*time[a:b]), -.999, .999)
    stream = BytesIO()
    sf.write(stream, samples, rate, subtype="PCM_16", format="WAV")
    return stream.getvalue()


def generated_entries():
    ensure_sample_library()
    with sqlite3.connect(ROOT / "data" / "signaltrace-library.sqlite") as conn:
        rows = conn.execute("SELECT sample_id, title, duration, extension FROM sample_audio ORDER BY sample_id").fetchall()
    return [{"id": rid, "label": title, "source": "generated_signal",
             "speech": False, "seconds": duration, "extension": ext}
            for rid, title, duration, ext in rows]


def ensure_sample_library():
    """Persistent local SQLite index of four reproducible, clearly labeled fixtures."""
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    folder = data / "signaltrace-samples"
    folder.mkdir(parents=True, exist_ok=True)
    db = data / "signaltrace-library.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sample_audio (sample_id TEXT PRIMARY KEY, title TEXT NOT NULL, duration REAL NOT NULL, extension TEXT NOT NULL, path TEXT NOT NULL)")
        for i, title in enumerate(("Changing frequencies", "Level and silence",
                                   "Spectral transitions", "Extended signal check")):
            sample = folder / f"signal-{i+1}.wav"
            if not sample.is_file():
                sample.write_bytes(_generated(i))
            conn.execute("INSERT OR IGNORE INTO sample_audio VALUES (?,?,?,?,?)",
                         (f"signal-{i+1}", title, (16,23,31,42)[i], ".wav", str(sample)))


def local_recordings():
    # Databases catalog file paths; audio bytes aren't stored in SQLite.
    # Never auto-create the old demo database or download external media.
    result = []
    data = ROOT / "data"
    if not data.exists():
        return result
    for db in sorted(data.glob("*.sqlite"))[:8]:
        try:
            connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
            try:
                rows = connection.execute(
                    "SELECT id, path, kind, duration FROM recordings ORDER BY id LIMIT 300"
                ).fetchall()
            finally:
                connection.close()
            for rid, filename, kind, duration in rows:
                p = Path(filename)
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                # Only files inside this local project. Don't surface hidden personal paths.
                if not p.is_relative_to(ROOT) or not p.is_file() or p.suffix.lower() not in {
                    ".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"
                }:
                    continue
                if not 0 < p.stat().st_size <= 256*1024*1024:
                    continue
                result.append({"id": f"{db.stem}:{rid}", "label": f"Catalog clip {len(result)+1:03d}",
                               "source": "existing_catalog", "speech": None,
                               "seconds": duration, "extension": p.suffix.lower(), "_path": p})
                if len(result) >= 120:
                    return result
        except (sqlite3.Error, OSError, ValueError):
            continue
    return result


def catalog():
    return generated_entries() + [{k:v for k,v in x.items() if k != "_path"}
                                  for x in local_recordings()]


def choose():
    existing = local_recordings()
    # Prefer existing locally available audio when present; it may be synthetic.
    choices = existing if existing else generated_entries()
    return random.SystemRandom().choice(choices)


def load(entry):
    if entry["source"] == "generated_signal":
        ensure_sample_library()
        with sqlite3.connect(ROOT / "data" / "signaltrace-library.sqlite") as conn:
            row = conn.execute("SELECT path FROM sample_audio WHERE sample_id=?", (entry["id"],)).fetchone()
        if row is None:
            raise ValueError("Unknown sample")
        resolved = Path(row[0]).resolve()
        allowed = (ROOT / "data" / "signaltrace-samples").resolve()
        if not resolved.is_relative_to(allowed):
            raise ValueError("Sample path outside catalog")
        return resolved.read_bytes()
    if entry["source"] == "existing_catalog":
        return Path(entry["_path"]).read_bytes()
    raise ValueError("Unknown sample source")


def resolve(identifier):
    for entry in [*generated_entries(), *local_recordings()]:
        if entry["id"] == identifier:
            return entry
    return None
