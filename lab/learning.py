"""Optional reference examples, kept outside the assigned blind study."""
from pathlib import Path
import random
import tarfile


def examples(store):
    wanted = set(store.setting("learning_examples", []))
    if not wanted:
        return []
    # Check both recording and source group, even if a setting is stale.
    marks = ",".join("?" for _ in wanted)
    rows = store.rows(f"""SELECT r.id,l.label,r.codec FROM recordings r
        JOIN reference_labels l ON l.recording_id=r.id WHERE r.id IN ({marks}) AND r.subset='progress'
        AND r.source_group IS NOT NULL
        AND r.source_group NOT IN (SELECT x.source_group FROM assignments a
             JOIN recordings x ON x.id=a.recording_id WHERE x.source_group IS NOT NULL)
        ORDER BY r.trial_id""", tuple(wanted))
    by_id = {r["id"]: r for r in rows if r["id"] in wanted}
    return [by_id[key] for key in store.setting("learning_examples", []) if key in by_id]


def prepare_examples(store, archive):
    if store.setting("dataset_kind") != "research" or store.setting("reviewer_mode"):
        raise ValueError("Reference examples require an owner-side research workspace")
    existing = examples(store)
    if existing:
        return existing
    rows = store.rows("""SELECT r.*,l.label FROM recordings r JOIN reference_labels l ON l.recording_id=r.id
        WHERE r.subset='progress' AND r.source_group IS NOT NULL
        AND r.source_group NOT IN (SELECT x.source_group FROM assignments a
             JOIN recordings x ON x.id=a.recording_id WHERE x.source_group IS NOT NULL)
        ORDER BY r.trial_id""")
    random.Random(20261019).shuffle(rows)
    candidates, groups, counts = [], set(), {"human": 0, "synthetic": 0}
    for r in rows:
        if r["source_group"] in groups or counts[r["label"]] >= 12:
            continue
        groups.add(r["source_group"])
        counts[r["label"]] += 1
        candidates.append(r)
        if len(candidates) == 24:
            break
    by_name = {r["trial_id"] + ".flac": r for r in candidates}
    created = set()
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in by_name:
                continue
            r = by_name[name]
            if not member.isfile() or member.size > 20_000_000:
                raise ValueError("Unexpected source audio member")
            path = Path(r["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(tar.extractfile(member).read())
                created.add(path)
    chosen, counts = [], {"human": 0, "synthetic": 0}
    for r in candidates:
        if counts[r["label"]] >= 2:
            continue
        try:
            store.validate_media(r["id"])
        except ValueError:
            continue
        counts[r["label"]] += 1
        chosen.append(r["id"])
    if len(chosen) != 4:
        raise ValueError("Could not prepare two readable examples per class")
    with store.conn:
        store.set_setting("learning_examples", chosen)
    keep = {Path(r["path"]) for r in candidates if r["id"] in chosen}
    for path in created - keep:
        path.unlink()
    return examples(store)


def forensic_examples(store):
    """Duration-qualified demonstrations, never selected by detector outcomes."""
    wanted = store.setting('forensic_examples', [])
    if not wanted:
        return []
    marks = ','.join('?' for _ in wanted)
    rows = store.rows(f'''SELECT r.id,l.label,r.codec,r.duration FROM recordings r
      JOIN reference_labels l ON l.recording_id=r.id
      WHERE r.id IN ({marks}) AND r.duration BETWEEN 2.5 AND 13 AND r.source_group IS NOT NULL
      AND r.source_group NOT IN (SELECT x.source_group FROM assignments a
        JOIN recordings x ON x.id=a.recording_id WHERE x.source_group IS NOT NULL)''',tuple(wanted))
    by_id={r['id']:r for r in rows}
    return [by_id[k] for k in wanted if k in by_id]
