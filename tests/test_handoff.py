import hashlib
from io import BytesIO
import json
import zipfile

import pytest

from lab.handoff import export_package, export_results, import_results, load_package
from lab.reporting import findings
from lab.store import Store
from conftest import payload, submit_phase


def test_independent_roundtrip_is_blind_atomic_and_idempotent(db, tmp_path):
    blob = export_package(db, "independent-person", "main-02", 3)
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        text = json.dumps(manifest)
        assert not any(key in text for key in ('"reference"', '"score"', '"source_group"', '"trial_id"', 'DEMO_'))
        assert set(archive.namelist()) == {"manifest.json", "ATTRIBUTION.txt"} | {r["media"] for r in manifest["assignments"]}
    independent = Store(tmp_path / "independent.sqlite")
    load_package(independent, blob)
    assert not independent.rows("SELECT * FROM reference_labels")
    assert not independent.rows("SELECT * FROM detector_scores")
    for a in independent.queue("independent-person"):
        independent.save(a["id"], payload(), "submitted")
    results = export_results(independent)
    submit_phase(db, "main-02")
    with pytest.raises(ValueError, match="submitted"):
        db.reveal_phase("main-02")
    broken = json.loads(results)
    broken["reviews"][-1]["annotation"]["confidence"] = 99
    with pytest.raises(ValueError, match="Confidence"):
        import_results(db, json.dumps(broken))
    assert not db.one("SELECT 1 FROM assignments WHERE role='independent' AND submitted_at IS NOT NULL")
    assert import_results(db, results) == 3
    assert import_results(db, results) == 0
    broken = json.loads(results)
    broken["reviews"][0]["annotation"]["rationale"] = "Conflicting result"
    with pytest.raises(ValueError, match="Conflicting"):
        import_results(db, json.dumps(broken))
    db.reveal_phase("main-02")
    assert findings(db)["independent"]["pairs"] == 3
    assert findings(db)["independent"]["agreement"] == 1
    with pytest.raises(ValueError, match="unrevealed"):
        export_package(db, "third-person", "main-02", 1)
    independent.close()


def test_duplicate_and_tampered_review_packages_rejected(db, tmp_path):
    blob = export_package(db, "other", "calibration", 2)
    reader = Store(tmp_path / "reader.sqlite")
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest["assignments"][1]["id"] = manifest["assignments"][0]["id"]
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, json.dumps(manifest).encode() if name == "manifest.json" else data)
    with pytest.raises(ValueError, match="duplicate"):
        load_package(reader, output.getvalue())
    assert not reader.rows("SELECT * FROM recordings")
    reader.close()
