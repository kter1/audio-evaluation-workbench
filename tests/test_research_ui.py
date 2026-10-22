from pathlib import Path
from streamlit.testing.v1 import AppTest
from lab.research import ResearchDB
import research_app

APP = str(Path(__file__).resolve().parents[1] / "research_app.py")

def test_overview_and_demo_import(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERACTION_LAB_DB", str(tmp_path / "research.sqlite"))
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    assert "Research quality" in app.title[0].value
    next(x for x in app.button if x.label == "Explore sample dataset").click().run()
    assert not app.exception

def test_audited_dataset_renders_investigation(monkeypatch, tmp_path):
    target = tmp_path / "research.sqlite"
    monkeypatch.setenv("INTERACTION_LAB_DB", str(target))
    from json import loads
    from pathlib import Path
    store = ResearchDB(target)
    manifest = loads((Path(__file__).resolve().parents[1] / "fixtures" / "demo_research.json").read_text())
    store.import_manifest(manifest)
    store.audit("demonstration-v1")
    store.close()
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    # Select Investigations using the sidebar radio widget
    next(r for r in app.sidebar.radio if r.label == "Section").set_value("Investigations").run()
    assert not app.exception
    assert "Investigate flagged issues" in [x.value for x in app.title]
