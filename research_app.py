"""Interaction Research Lab — local-first dataset QA and investigation interface."""
import json
import os
from pathlib import Path

import streamlit as st
from lab.research import ResearchDB, RULES, RULESET

ROOT = Path(__file__).resolve().parent
DB = Path(os.environ.get("INTERACTION_LAB_DB", str(ROOT / "data" / "research.sqlite")))
DEMO = ROOT / "fixtures" / "demo_research.json"
EVAL = ROOT / "docs" / "detector-evaluation.json"

st.set_page_config(page_title="Interaction Research Lab", page_icon="🔎", layout="wide")
st.markdown("<style>" + (ROOT / "assets/research_style.css").read_text() + "</style>", unsafe_allow_html=True)

def main():
    db = ResearchDB(DB)
    try:
        with st.sidebar:
            st.markdown('<div class="research-brand">◉ <span>Interaction Lab</span></div>', unsafe_allow_html=True)
            st.caption("RESEARCH QUALITY CONTROL")
            page = st.radio("Section", ["Overview", "Datasets", "Investigations", "Audio analysis", "Reports"])
            st.divider()
            st.caption("Offline-first · local single-operator software · no login or shared access.")
            st.caption("The demo consists of deliberately constructed metadata, not real Pindrop records.")
        ds = db.datasets()
        dataset_id = None
        if ds:
            choices = [x["id"] for x in ds]
            dataset_id = st.selectbox("Research dataset", choices, format_func=lambda x: next(d["name"] for d in ds if d["id"] == x))
        if page == "Overview":
            st.title("Research quality, with an evidence trail.")
            st.write("Import → audit → investigate → report. The platform checks annotation integrity and surfaces discrepancies without treating an AI model's predictions as verified truth.")
            if st.button("Explore sample dataset", type="primary"):
                try:
                    added = db.import_manifest(json.loads(DEMO.read_text()))
                    count = db.audit("demonstration-v1")
                    st.success(("Imported" if added else "Reused") + f" the transparent test fixture. {count} rule findings; no real speech or research participant data.")
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
            if dataset_id:
                d, records, annotations = db.manifest(dataset_id)
                issues = db.issues(dataset_id)
                cols = st.columns(4)
                for col, name, value in zip(cols, ["Recordings", "Annotation rows", "QA findings", "Open issues"],
                                            [len(records), len(annotations), len(issues), sum(x["status"] == "open" for x in issues)]):
                    col.metric(name, value)
                st.subheader("Issues detected")
                from collections import Counter
                st.dataframe([{"Issue": key.replace("_", " ").title(), "Findings": value, "Priority": {"error":"Data error","warning":"Check source","review":"Needs judgment"}[RULES[key][0]]}
                              for key, value in Counter(x["rule_id"] for x in issues).most_common()],
                             hide_index=True, use_container_width=True)
            else:
                st.info("Start by selecting Explore sample dataset. You can investigate deliberately seeded issues without any model downloads.")
        elif page == "Datasets":
            st.title("Your datasets")
            st.caption("Immutable versioned imports. Each manifest contains recording metadata and source annotations; model scores are separate.")
            with st.expander("Import research JSON manifest", expanded=not ds):
                upload = st.file_uploader("Manifest JSON (maximum 5 MB)", type=["json"])
                if upload is not None and st.button("Validate and import"):
                    try:
                        if upload.size > 5_000_000:
                            raise ValueError("Manifest exceeds 5 MB.")
                        incoming = json.loads(upload.getvalue())
                        added = db.import_manifest(incoming)
                        st.success("Dataset imported." if added else "Identical dataset already imported.")
                        st.rerun()
                    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as error:
                        st.error(str(error))
            if dataset_id:
                d, records, annotations = db.manifest(dataset_id)
                st.markdown(f"**Origin:** {d['origin']}")
                with st.expander("Source fingerprint"):
                    st.code(d["manifest_sha"], language=None)
                a, b = st.tabs(["Recording rows", "Annotation rows"])
                with a:
                    st.dataframe(records, hide_index=True, use_container_width=True)
                with b:
                    st.dataframe(annotations, hide_index=True, use_container_width=True)
                if st.button("Run deterministic QA audit", type="primary"):
                    st.success(f"Audit completed: {db.audit(dataset_id)} findings (ruleset {RULESET}).")
                    st.rerun()
        elif page == "Investigations":
            st.title("Investigate flagged issues")
            if not dataset_id:
                st.info("Import a dataset first.")
            else:
                issues = db.issues(dataset_id)
                if not issues:
                    st.info("No recorded findings. Run an audit under Datasets.")
                else:
                    left, right = st.columns([2, 3], gap="large")
                    with left:
                        severity = st.selectbox("Issue type", ["All", "error", "warning", "review"])
                        status = st.selectbox("Progress", ["All", "open", "investigating", "resolved", "dismissed", "escalated"])
                        rule = st.selectbox("Rule", ["All"] + sorted({r["rule_id"] for r in issues}))
                        filtered = [i for i in issues if (severity == "All" or i["severity"] == severity)
                                    and (status == "All" or i["status"] == status)
                                    and (rule == "All" or i["rule_id"] == rule)]
                        st.caption(f"{len(filtered)} of {len(issues)} findings")
                        if filtered:
                            selected = st.selectbox("Select a finding", [x["id"] for x in filtered], format_func=lambda key:
                                next(f"{r['severity'].upper()} · {r['rule_id']} · {r['recording_id'] or 'dataset'}" for r in filtered if r["id"] == key))
                        else:
                            selected = None
                    with right:
                        if selected:
                            record = db.issue(selected)
                            st.subheader(record["rule_id"].replace("_", " ").title())
                            st.caption(f"{record['severity'].upper()} · {record['status']} · issue {selected}")
                            st.markdown("**What triggered this finding?**")
                            st.dataframe([{"Detail": k.replace("_", " ").title(), "Value": str(v)} for k,v in record["evidence"].items()], hide_index=True, use_container_width=True)
                            st.markdown("**Recommended next check**")
                            st.write(record["next_check"])
                            d, recordings, ann = db.manifest(dataset_id)
                            matching = [r for r in recordings if r.get("recording_id") == record["recording_id"]]
                            matching_ann = [a for a in ann if a.get("recording_id") == record["recording_id"]]
                            if matching:
                                with st.expander("Source recording metadata"):
                                    st.dataframe(matching, hide_index=True, use_container_width=True)
                            if matching_ann:
                                with st.expander("Original imported annotations"):
                                    st.dataframe(matching_ann, hide_index=True, use_container_width=True)
                            st.info("Reference data, imported annotations, and detector predictions are distinct. A flag is not automatically a confirmed annotation error.")
                            st.subheader("Decision history")
                            if record["events"]:
                                st.dataframe(record["events"], hide_index=True, use_container_width=True)
                            else:
                                st.caption("No dispositions recorded.")
                            with st.form("disposition_" + selected):
                                result = st.selectbox("Outcome", ["investigating", "resolved", "dismissed", "escalated", "open"])
                                actor = st.text_input("Analyst ID", value="analyst")
                                explanation = st.text_area("Reason or evidence")
                                if st.form_submit_button("Save decision"):
                                    try:
                                        db.disposition(selected, result, actor, explanation)
                                        st.rerun()
                                    except ValueError as error:
                                        st.error(str(error))
        elif page == "Audio analysis":
            st.title("Audio analysis")
            st.write("Archived model results are previous experiments, **not inference on newly uploaded recordings**. Optional full detector inference remains available in the legacy app.")
            if EVAL.exists():
                try:
                    report = json.loads(EVAL.read_text())
                    st.caption(f"Recorded evaluation version: {report.get('version', 'unknown')}. See docs/model-evaluation.md for sampling methodology and limitations.")
                    st.dataframe([{"Condition": {"clean":"Clean audio","mp3":"MP3 compression","phone":"Telephone audio","noise20":"Background noise"}.get(k,k), "Synthetic detected": f"{v[\'fakes\'][\'flagged_synthetic\']}/{v[\'fakes\'][\'n\']}", "Human falsely flagged": f"{v[\'reals\'][\'flagged_synthetic\']}/{v[\'reals\'][\'n\']}"} for k,v in report.get("test",{}).get("by_channel",{}).items()], hide_index=True, use_container_width=True)
                    st.download_button("Download archived evaluation receipt", EVAL.read_bytes(), "archived-detector-evaluation.json", "application/json")
                except (ValueError, OSError) as error:
                    st.error(str(error))
            st.divider()
            st.subheader("Inspect an audio recording")
            file = st.file_uploader("Local WAV/FLAC/MP3/OGG/AIFF", type=["wav","flac","mp3","ogg","aiff","aif"])
            if file is not None:
                try:
                    from lab.forensics import read_audio, diagnostics
                    blob = file.getvalue()
                    audio, rate, details = read_audio(blob)
                    st.audio(blob)
                    st.metric("Length", f"{details[\'duration_seconds\']:.1f} s")
                    _, metrics, warnings = diagnostics(audio, rate)
                    st.subheader("Measured signal properties")
                    st.dataframe([{"Measurement": k.replace("_", " ").title(), "Value": str(v)} for k,v in metrics.items()], hide_index=True, use_container_width=True)
                    for warning in warnings:
                        st.warning(warning)
                    st.caption("Descriptive signal measurements are not a speech-origin classification.")
                    # The legacy forensics module exports rich signal and detector workflows;
                    # use only validated media here, without installing inference checkpoints.
                except (ValueError, RuntimeError, OSError) as error:
                    st.error(str(error))
            st.caption("For full local model inference run ./run-legacy.sh and choose Detector Lab. Some model licenses restrict commercial use.")
        elif page == "Reports":
            st.title("Research reports")
            if not dataset_id:
                st.info("Import and audit a dataset first.")
            else:
                d, records, anns = db.manifest(dataset_id)
                issues = db.issues(dataset_id)
                st.metric("Research QA flags", len(issues))
                st.write("The export preserves source dataset provenance, rule versions, issue evidence, dispositions, and limitations. No private recordings are included.")
                st.download_button("Download research report", db.export_bundle(dataset_id),
                                   file_name=f"{dataset_id}-research-report.zip", mime="application/zip", type="primary")
                st.caption("The demonstration fixture contains injected defects; its QA counts are not accuracy claims on real customer data.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
