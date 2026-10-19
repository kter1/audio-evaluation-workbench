from lab.learning import examples


def test_learning_examples_cannot_expose_assigned_recordings_or_sources(db):
    assigned = db.one("SELECT recording_id FROM assignments LIMIT 1")["recording_id"]
    safe = db.one("SELECT id FROM recordings WHERE subset='progress' LIMIT 1")["id"]
    with db.conn:
        db.set_setting("learning_examples", [assigned, safe])
    result = examples(db)
    assert [r["id"] for r in result] == [safe]
    assert set(result[0]) == {"id", "label", "codec"}
    assert not db.rows("SELECT * FROM annotation_revisions")
    # Even a progress recording is excluded if its source group is assigned.
    with db.conn:
        db.conn.execute("UPDATE recordings SET source_group=(SELECT source_group FROM recordings WHERE id=?) WHERE id=?", (assigned, safe))
    assert examples(db) == []
