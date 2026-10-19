"""Audio Review Lab: a guided listening and research workspace."""
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab.audio import wav_bytes
from lab.forensics_ui import forensics_screen
from lab.handoff import export_package, export_results, import_results, package_bytes
from lab.learning import examples
from lab.reporting import (adjudications_csv, annotation_csv, calibration_feedback, cases_csv, challenge_summary, cue_analysis,
                           csv_bytes, findings, history_csv, report_html, research_bundle, review_rows)
from lab.store import CUES, HUMAN_CUES, INTELLIGIBILITY, JUDGMENTS, PHASES, SYNTHETIC_CUES, TAGS, Store, now
from lab.workflow import PHASE_LABELS, progress

st.set_page_config(page_title="Audio Review Lab", page_icon="◉", layout="wide")
st.markdown("<style>" + (Path(__file__).parent / "assets/style.css").read_text() + "</style>", unsafe_allow_html=True)


WORKSPACES = {"data/study.sqlite": "Research study", "data/challenge-v4.sqlite": "Listening challenge", "data/demo.sqlite": "Practice demo"}


def default_workspace():
    return os.environ.get("AUDIO_LAB_DB", "data/demo.sqlite")


def store():
    return Store(st.session_state.get("workspace_db", default_workspace()))


def workspace_switcher():
    options = list(dict.fromkeys([default_workspace(), *(p for p in WORKSPACES if Path(p).exists())]))
    if st.session_state.get("workspace_db") not in options:
        st.session_state["workspace_db"] = options[0]
    if len(options) > 1:
        st.selectbox("Workspace", options, key="workspace_db", format_func=lambda p: WORKSPACES.get(p, Path(p).name))


def navigate(screen, phase=None):
    st.session_state["nav_request"] = (screen, phase)
    st.rerun()


def act(action, success):
    try:
        result = action()
        st.success(success)
        return result
    except (ValueError, OSError, KeyError) as error:
        st.error(str(error))
        return None


def fmt(value):
    return "Not measured" if value is None else f"{value:.1%}"


def heading(title, eyebrow, description):
    st.markdown(f'<p class="eyebrow">{escape(eyebrow)}</p>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def learning_panel(db):
    lessons = examples(db)
    if not lessons:
        return
    if not st.toggle("Warm up with reference examples", help="Four real clips outside your study sample. No review results are recorded."):
        return
    with st.container(border=True):
        st.markdown("#### Listen, then reveal the reference")
        st.caption("Practice only · these source groups are excluded from your blind study. Listening alone may be inconclusive.")
        indices = list(range(len(lessons)))
        index = st.selectbox("Reference example", indices, format_func=lambda i: f"Example {i + 1}")
        item = lessons[index]
        try:
            audio, rate = db.validate_media(item["id"])
            st.audio(wav_bytes(audio, rate), format="audio/wav")
        except ValueError as error:
            st.error(str(error))
            return
        impression = st.radio("Your practice impression", JUDGMENTS, index=2, horizontal=True,
                              format_func=str.title, key="practice_impression_" + item["id"])
        if st.button("Reveal this reference", key="reveal_practice_" + item["id"]):
            st.session_state["practice_revealed_" + item["id"]] = True
        if st.session_state.get("practice_revealed_" + item["id"]):
            st.info(f"Dataset reference: {item['label'].title()} · transmission codec: {item['codec']}")
            st.write("Your impression matched the reference." if impression == item["label"] else
                     "The reference establishes provenance. An uncertain or different listening impression is useful evidence about the limits of listening alone.")
            st.caption("Describe audible artifacts precisely. Noise, compression, or unusual prosody alone do not establish synthesis or fraud.")


def overview_screen(db, reviewer):
    state = progress(db, reviewer)
    heading("Your review workspace", "AUDIO / RESEARCH / EVIDENCE",
            "A guided path from careful listening to findings you can explain.")
    step = state["next"]
    st.markdown(f'<div class="hero"><p class="eyebrow">YOUR NEXT STEP</p><h2>{escape(step["label"])}</h2><p>{escape(step["description"])}</p></div>', unsafe_allow_html=True)
    if st.button(step["label"], type="primary", key="next_step"):
        navigate(step["screen"], step["phase"])
    learning_panel(db)
    a, b, c = st.columns(3)
    a.metric("Recordings ready", state["total"])
    b.metric("Personally reviewed", state["done"])
    case_count = db.one("SELECT COUNT(*) AS n FROM cases")["n"]
    c.metric("Case studies written", case_count)
    st.subheader("Your listening challenge" if db.setting("dataset_kind") == "challenge" else "Your study, in four steps")
    descriptions = {
        "calibration": "Learn the guide. Compare this batch with reference labels before your main study.",
        "main-01": "Build the first portfolio milestone. Then check consistency with blind repeats.",
        "main-02": "Expand the evidence while repeat reviews wait for their 48-hour interval.",
        "validation": "Apply your revised guide to fresh recordings. Compare the findings descriptively.",
        "challenge": "Twenty clips: real people and modern voice clones. Judge each blind, then see which fooled you and whether the detector caught them.",
    }
    for number, batch in enumerate(state["batches"], 1):
        with st.container(border=True, key="stage_" + batch["phase"]):
            a, b = st.columns([4, 1])
            a.markdown(f'<div class="step-label">STEP {number:02d}</div>', unsafe_allow_html=True)
            a.markdown(f"**{batch['label']}**")
            a.caption(descriptions[batch["phase"]])
            b.markdown(f"**{batch['done']} / {batch['total']}**")
            st.progress(batch["done"] / batch["total"])
            status = "Comparison available" if batch["revealed"] else ("Ready for quality checks" if batch["done"] == batch["total"] else f"{batch['total'] - batch['done']} recordings remaining")
            st.caption(status + (f" · {batch['drafts']} saved drafts" if batch["drafts"] else ""))
            if st.button("Open comparison" if batch["revealed"] else "Open batch", key="open_" + batch["phase"]):
                navigate("Quality Review" if batch["revealed"] else "Review", batch["phase"])
    st.markdown('<div class="helper"><strong>What you get out of this</strong><br>Completed listening records → consistency checks → investigated detector mistakes → a report and evidence bundle. Your observations and conclusions are the work this project demonstrates.</div>', unsafe_allow_html=True)
    with st.expander("How to use this workspace"):
        st.markdown("**1. Listen** at the original speed. **2. Decide** using the guide; uncertainty is valid. **3. Document** what you heard and submit. After the batch is complete, Quality Review unlocks comparisons. Calibration feedback appears after the first batch; it stays separate from your study results.")
        st.caption("Independent review and research-case writing can be managed here, without terminal commands.")


def annotation_form(db, aid, audio, rate, initial=None, correction=False, next_aid=None):
    initial = initial or {"intelligibility": "clear", "tags": [], "cues": [], "judgment": "uncertain", "confidence": 3,
                          "observations": [], "rationale": "", "escalated": False, "escalation_reason": ""}
    with st.form("annotation_" + aid + ("_correction" if correction else "")):
        st.markdown("#### Your assessment")
        prefix = aid + ("_correction" if correction else "_original")
        judgment = st.radio("Perceived origin", JUDGMENTS, index=JUDGMENTS.index(initial["judgment"]),
                            horizontal=True, format_func=str.title, key=prefix + "_judgment",
                            help="A listening impression. Reference labels remain hidden until this batch is revealed.")
        left, right = st.columns([1, 1])
        intelligibility = left.selectbox("Intelligibility", INTELLIGIBILITY, index=INTELLIGIBILITY.index(initial["intelligibility"]), key=prefix + "_intelligibility")
        confidence = right.slider("Confidence", 1, 5, initial["confidence"], key=prefix + "_confidence",
                                  help="1 = weak evidence; 5 = strong evidence. This is self-reported, not a probability.")
        tags = st.multiselect("Audible interference", TAGS, default=initial["tags"], key=prefix + "_tags", placeholder="None, or select what you heard")
        cues = st.multiselect("What you heard (perceptual cues)", CUES, default=initial.get("cues", []), key=prefix + "_cues",
                              placeholder="Select every cue you actually heard",
                              format_func=lambda c: ("↑ synthetic · " if c in SYNTHETIC_CUES else "↑ human · " if c in HUMAN_CUES else "") + c,
                              help="Record cues before reveal. Afterwards, Findings shows which cues actually predicted the truth. Cues are evidence, not verdicts.")
        rationale = st.text_area("Decision notes", initial["rationale"], height=90,
                                 placeholder="What supports your judgment? What remains unclear?",
                                 help="Required when origin is uncertain.", key=prefix + "_rationale")
        with st.expander("Timestamped observations", expanded=bool(initial["observations"])):
            st.caption("Add the exact interval and a specific observation. Times refer to the full recording.")
            observations = st.data_editor(pd.DataFrame(initial["observations"], columns=["start", "end", "note"]),
                                         num_rows="dynamic", hide_index=True, width="stretch",
                                         column_config={"start": st.column_config.NumberColumn("Start (s)", min_value=0, max_value=len(audio)/rate, step=0.01),
                                                        "end": st.column_config.NumberColumn("End (s)", min_value=0, max_value=len(audio)/rate, step=0.01),
                                                        "note": st.column_config.TextColumn("What did you hear?")},
                                         key="notes_" + aid + str(correction))
        with st.expander("Flag for a second look", expanded=initial["escalated"]):
            escalated = st.checkbox("Escalate for further review", initial["escalated"], key=prefix + "_escalated")
            escalation_reason = st.text_area("Escalation reason", initial["escalation_reason"], key=prefix + "_escalation", height=80,
                                            placeholder="What question should another reviewer investigate?")
        skip = False
        if correction:
            draft = False
            submit = st.form_submit_button("Save separate correction", type="primary", width="stretch")
        else:
            submit = st.form_submit_button("Submit & next", type="primary", width="stretch",
                                           help="Freezes this original judgment and opens the next recording.")
            a, b = st.columns(2)
            draft = a.form_submit_button("Save draft", width="stretch")
            skip = b.form_submit_button("Save & skip", width="stretch", disabled=next_aid is None)
            st.caption("Drafts can be edited. Submitted judgments stay frozen; later corrections are separate.")
    if draft or submit or skip:
        notes = []
        for row in observations.to_dict("records"):
            if all(pd.isna(v) or v == "" for v in row.values()):
                continue
            notes.append({"start": row["start"], "end": row["end"], "note": row["note"]})
        payload = {"intelligibility": intelligibility, "tags": tags, "cues": cues, "judgment": judgment, "confidence": confidence,
                   "observations": notes, "rationale": rationale, "escalated": escalated, "escalation_reason": escalation_reason}
        try:
            db.save(aid, payload, "correction" if correction else ("submitted" if submit else "draft"))
            st.session_state["flash"] = "Correction recorded; original preserved." if correction else ("Assessment submitted. Ready for the next clip." if submit else "Draft saved. You can return to it at any time.")
            if not correction and (submit or skip):
                st.session_state["review_next"] = next_aid
                st.session_state.pop("review_selection", None)
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


def player(audio, rate, aid):
    duration = len(audio) / rate
    step = max(1, len(audio) // 4000)
    figure = go.Figure(go.Scatter(x=np.arange(0, len(audio), step) / rate,
                                  y=audio[::step].mean(axis=1), mode="lines",
                                  fill="tozeroy", fillcolor="rgba(230,107,76,.12)",
                                  line={"color": "#dc765b", "width": 1.2}))
    figure.update_layout(height=190, margin={"l": 0, "r": 0, "t": 14, "b": 24},
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         xaxis={"title": None, "ticksuffix": "s", "gridcolor": "#edf0f4", "zeroline": False},
                         yaxis={"visible": False}, showlegend=False, dragmode=False, font={"color": "#8390a1", "size": 11})
    st.plotly_chart(figure, use_container_width=True, key="wave_" + aid, config={"displayModeBar": False})
    st.audio(wav_bytes(audio, rate), format="audio/wav")
    st.caption(f"{duration:.2f} seconds · {rate:,} Hz · original speed")
    with st.expander("Replay a precise interval"):
        a, b = st.columns(2)
        start = a.number_input("Segment start (s)", 0.0, duration, 0.0, step=0.01, key="start_" + aid)
        end = b.number_input("Segment end (s)", 0.0, duration, min(duration, 1.0), step=0.01, key="end_" + aid)
        loop = st.checkbox("Loop this interval", key="loop_" + aid)
        if end > start:
            st.audio(wav_bytes(audio, rate, start, end), format="audio/wav", loop=loop)
        else:
            st.info("Choose an end after the start.")


def review_screen(db, reviewer):
    heading("Review", "THE LISTENING DESK", "Listen first. Record your judgment. Check the evidence after the batch.")
    state = progress(db, reviewer)
    queue = db.queue(reviewer)
    if not queue:
        st.info("No assignments yet. The setup guide explains how to prepare your workspace.")
        return
    phases = [phase for phase in PHASES if any(a["phase"] == phase for a in queue)]
    if st.session_state.get("batch_selector") not in phases:
        st.session_state["batch_selector"] = phases[0]
    phase = st.selectbox("Batch", phases, format_func=lambda p: PHASE_LABELS[p], key="batch_selector")
    items = [a for a in queue if a["phase"] == phase]
    completed = sum(bool(a["submitted_at"]) for a in items)
    st.progress(completed / len(items), text=f"{completed} of {len(items)} submitted · {len(items) - completed} remaining")
    pending = [a for a in items if not a["submitted_at"]]
    ready = [a for a in pending if not a["due_at"] or a["due_at"] <= now()]
    if not ready:
        if pending:
            due = min(a["due_at"] for a in pending)
            local = datetime.fromisoformat(due).astimezone(ZoneInfo("America/New_York"))
            st.info(f"Repeat reviews open {local.strftime('%b %d at %I:%M %p')} Eastern. You can work on another batch meanwhile.")
        else:
            st.success("This batch is submitted. Your next step is to check the comparisons.")
            if st.button("Check this batch", type="primary"):
                navigate("Quality Review", phase)
        return
    by_id = {a["id"]: a for a in ready}
    queued = st.session_state.pop("review_next", None)
    if queued in by_id:
        st.session_state["review_selection"] = queued
    if st.session_state.get("review_selection") not in by_id:
        st.session_state["review_selection"] = next((a["id"] for a in ready if a["id"] in state["draft_ids"]), ready[0]["id"])
    numbers = {a["id"]: i + 1 for i, a in enumerate(items)}
    aid = st.selectbox("Recording", list(by_id), key="review_selection",
                      format_func=lambda key: f"Clip {numbers[key]:02d}" + (" · saved draft" if key in state["draft_ids"] else "") + (" · blind repeat" if by_id[key]["role"] == "repeat" else ""))
    next_aid = next((a["id"] for a in ready if a["id"] != aid and numbers[a["id"]] > numbers[aid]), None)
    if next_aid is None:
        next_aid = next((a["id"] for a in ready if a["id"] != aid), None)
    record = db.blind_review(aid)
    try:
        audio, rate = db.review_audio(aid)
    except ValueError as error:
        st.error(str(error))
        st.caption("This assignment has not been completed. Resolve the media issue and retry.")
        return
    listen, assess = st.columns([1.1, 1], gap="large")
    with listen:
        with st.container(border=True, key="listening_panel"):
            st.markdown(f'<div class="step-label">01 / LISTEN</div><h3>Clip {numbers[aid]:02d}</h3><div class="clip-meta">Blind review · guideline {escape(record["guideline"])}</div>', unsafe_allow_html=True)
            player(audio, rate, aid)
        st.markdown('<div class="helper"><strong>Listen for evidence, not a hunch.</strong><br>First assess clarity and interference. Then judge perceived origin. If evidence is insufficient, choose uncertain and explain why.</div>', unsafe_allow_html=True)
        with st.expander("Keep the annotation guide nearby"):
            st.markdown(record["guide"])
        st.caption("After calibration, you can compare this batch with the reference labels and inspect where you disagreed.")
    with assess:
        annotation_form(db, aid, audio, rate, record["annotation"], next_aid=next_aid)


def independent_tools(db, phase):
    with st.expander("Get an independent review"):
        st.caption("Prepare a blind package for someone who has agreed to review. Their assigned reviews must return before this batch can be revealed.")
        phase_total = db.one("SELECT COUNT(*) AS n FROM assignments WHERE phase=? AND role='primary'", (phase,))["n"]
        with st.form("make_package"):
            name = st.text_input("Independent reviewer name", placeholder="A different person")
            count = st.number_input("Recordings to assign", min_value=1, max_value=max(1, phase_total),
                                    value=min(20 if phase == "main-01" else 30, max(1, phase_total)))
            create = st.form_submit_button("Prepare blind package")
        if create:
            try:
                export_package(db, name, phase, count)
                st.success("Package prepared. Download it below and share it with your reviewer.")
            except (ValueError, OSError) as error:
                st.error(str(error))
        packages = db.rows("SELECT id,manifest,results_hash FROM packages")
        for package in packages:
            manifest = json.loads(package["manifest"])
            if not manifest["assignments"] or manifest["assignments"][0]["phase"] != phase:
                continue
            label = manifest["reviewer"]
            st.caption(f"{label} · {len(manifest['assignments'])} clips · " + ("results returned" if package["results_hash"] else "awaiting results"))
            try:
                cache_key = f"package_download:{db.path}:{package['id']}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = package_bytes(db, package["id"])
                blob = st.session_state[cache_key]
                st.download_button(f"Download package for {label}", blob, f"blind_review_{package['id'][:8]}.zip", "application/zip", key="package_" + package["id"])
            except (ValueError, OSError) as error:
                st.error(str(error))
        upload = st.file_uploader("Return reviewer results", type=["json"])
        if upload is not None and st.button("Import independent reviews"):
            act(lambda: import_results(db, upload.getvalue()), "Independent reviews imported.")


def quality_screen(db, reviewer):
    heading("Quality Review", "CHECK / COMPARE / INVESTIGATE", "Understand disagreements and turn interesting cases into research findings.")
    if db.setting("reviewer_mode"):
        st.info("Your workspace contains independent assignments only. Submit them all, then return the results to the study owner.")
        try:
            st.download_button("Download completed reviews", export_results(db), "reviewer_results.json", "application/json", type="primary")
        except ValueError as error:
            st.caption(str(error))
        return
    challenge = db.setting("dataset_kind") == "challenge"
    phases = [p for p in PHASES if db.one("SELECT 1 FROM assignments WHERE phase=? LIMIT 1", (p,))] or list(PHASES)
    if st.session_state.get("quality_phase") not in phases:
        st.session_state["quality_phase"] = phases[0]
    phase = st.selectbox("Batch to manage", phases, format_func=lambda p: PHASE_LABELS[p], key="quality_phase")
    assignments = db.rows("SELECT role,submitted_at,revealed_at FROM assignments WHERE phase=?", (phase,))
    remaining = sum(not a["submitted_at"] for a in assignments)
    revealed = bool(assignments) and all(a["revealed_at"] for a in assignments)
    if not revealed:
        with st.container(border=True):
            st.markdown("#### Close this blind batch")
            if remaining:
                st.caption(f"{remaining} assigned reviews remain before comparisons can be unlocked.")
                if st.button("Continue listening", type="primary"):
                    navigate("Review", phase)
            elif not assignments:
                st.caption("No assignments in this batch yet.")
            elif challenge:
                st.caption("All clips are judged. Revealing shows the truth and the detector's frozen answers; your judgments stay frozen.")
            else:
                st.caption("All current assignments are submitted. Assign any intended independent review before revealing.")
            repeats = [a for a in assignments if a["role"] == "repeat"]
            primary_done = bool(assignments) and all(a["submitted_at"] for a in assignments if a["role"] == "primary")
            if phase == "main-01" and primary_done and not repeats:
                count = min(20, sum(a["role"] == "primary" for a in assignments))
                if st.button(f"Schedule {count} blind repeats"):
                    act(lambda: db.schedule_repeats(phase, count), "Repeats scheduled. They open 48 hours after their original submissions.")
            confirmed = challenge or st.checkbox("I have assigned all intended independent reviews")
            if st.button("Reveal batch comparison", type="primary", disabled=not confirmed or remaining > 0 or not assignments):
                try:
                    db.reveal_phase(phase)
                    st.session_state["flash"] = "Comparison unlocked. Your original judgments remain preserved."
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        if not challenge:
            independent_tools(db, phase)
    else:
        st.success("This batch is revealed. Original judgments are preserved; corrections and adjudications are separate.")
    with st.expander("Revise the guide for an unstarted batch"):
        st.caption("Versioned guides let you document what changed. A batch that already has any saved work cannot change its guide.")
        with st.form("guide_revision"):
            version = st.text_input("New guideline version", placeholder="2.0")
            guide_file = st.file_uploader("Revised guide", type=["md", "txt"])
            if st.form_submit_button("Register guide for this batch"):
                if guide_file is None:
                    st.error("Upload the revised guide first.")
                else:
                    try:
                        db.add_guideline(version, guide_file.getvalue().decode("utf-8"), phase)
                        st.success(f"Guide {version} registered for {PHASE_LABELS[phase]}.")
                    except (ValueError, UnicodeError) as error:
                        st.error(str(error))
    all_rows = review_rows(db)
    primary = [r for r in all_rows if r["role"] == "primary" and r["phase"] == phase]
    if not primary:
        st.markdown('<div class="helper"><strong>Your first useful checkpoint</strong><br>Finish calibration, reveal that batch, and inspect the comparison here. You do not need to complete the whole study to get feedback.</div>', unsafe_allow_html=True)
        return
    st.subheader("Find a case worth investigating")
    if phase == "calibration":
        st.caption("Calibration feedback only. These records are excluded from your main-study performance.")
    mode = st.selectbox("Focus", ["All reviewed cases", "Detector mistakes", "Reviewer disagreements", "Escalations", "Uncertain judgments"])
    def matches(r):
        related = [p for p in all_rows if p["recording_id"] == r["recording_id"] and p["role"] != "primary"]
        return (mode == "All reviewed cases"
                or mode == "Detector mistakes" and r["prediction"] is not None and r["prediction"] != r["reference"]
                or mode == "Reviewer disagreements" and any(p["judgment"] != r["judgment"] for p in related)
                or mode == "Escalations" and r["escalated"]
                or mode == "Uncertain judgments" and r["judgment"] == "uncertain")
    choices = [r for r in primary if matches(r)]
    if not choices:
        st.info("No revealed cases match this filter. That is a result, not a reason to manufacture an error.")
        return
    by_id = {r["assignment_id"]: r for r in choices}
    aid = st.selectbox("Case", list(by_id), format_func=lambda key: by_id[key]["trial_id"])
    r = by_id[aid]
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Evidence side by side")
        a, b, c = st.columns(3)
        a.metric("You heard", r["judgment"].title())
        b.metric("Dataset reference", r["reference"].title())
        c.metric("Detector prediction", r["prediction"].title() if r["prediction"] else "Unavailable")
        try:
            audio, rate = db.review_audio(aid)
            st.audio(wav_bytes(audio, rate), format="audio/wav")
            st.caption(f"{'Detector score' if challenge else 'Published score'} {r['score']} · {'generator' if challenge else 'codec'} {r['attack'] if challenge else r['codec']} · guide {r['guideline']}")
            st.write(r["rationale"] or "No decision notes recorded.")
            if r["observations"]:
                st.dataframe(pd.DataFrame(r["observations"]), hide_index=True)
            with st.expander("Record a separate correction"):
                annotation_form(db, aid, audio, rate, r, correction=True)
        except ValueError as error:
            st.error(str(error))
        related = [p for p in all_rows if p["recording_id"] == r["recording_id"]]
        if len(related) > 1:
            st.dataframe(pd.DataFrame([{k: p[k] for k in ("reviewer", "role", "judgment", "rationale")} for p in related]), hide_index=True)
        with st.expander("Adjudicate a disagreement"):
            with st.form("adjudication"):
                decision = st.selectbox("Adjudicated decision", JUDGMENTS)
                rationale = st.text_area("Evidence and adjudication rationale")
                if st.form_submit_button("Save adjudication"):
                    act(lambda: db.adjudicate(r["recording_id"], reviewer, decision, rationale), "Adjudication saved separately.")
            recorded = db.rows("SELECT reviewer,decision,rationale FROM adjudications WHERE recording_id=?", (r["recording_id"],))
            if recorded:
                st.dataframe(pd.DataFrame(recorded), hide_index=True)
    with right:
        with st.form("case"):
            st.markdown("#### Turn this into a research finding")
            title = st.text_input("Case title", placeholder="A concise description of the issue")
            observation = st.text_area("What the evidence shows", placeholder="Include timestamps, reference label, and detector behavior.")
            hypothesis = st.text_area("Possible explanation", placeholder="Separate what you know from what needs testing.")
            follow = st.text_area("Recommended next check", placeholder="What specific evidence or experiment would resolve this?")
            if st.form_submit_button("Save research case", type="primary", width="stretch"):
                act(lambda: db.add_case(r["recording_id"], title, observation, hypothesis, follow), "Research case saved to your handoff.")
        st.caption("Five well-investigated cases are the centerpiece of the portfolio. An unresolved case is valid when you explain what is missing.")


def feedback_metrics(metrics):
    a, b, c = st.columns(3)
    a.metric("Reviewed", metrics["reviewed"])
    b.metric("Uncertain", fmt(metrics["uncertainty_rate"]))
    c.metric("Accuracy among decided", fmt(metrics["accuracy_among_decided"]))


def challenge_panel(db, state):
    result = challenge_summary(review_rows(db))
    if not result["clips"]:
        st.markdown('<div class="hero"><p class="eyebrow">YOU VS. THE DETECTOR</p><h2>Judge every clip blind first.</h2><p>Listen to all twenty clips in Review, then reveal the batch in Quality Review. The detector already answered before you started; its scores stay hidden until then.</p></div>', unsafe_allow_html=True)
        if st.button(state["next"]["label"], type="primary"):
            navigate(state["next"]["screen"], state["next"]["phase"])
        return
    st.subheader("You vs. the detector")
    a, b, c, d = st.columns(4)
    a.metric("Fakes that fooled you", f"{result['fooled_you']} / {result['fakes']}",
             help="Judged human or uncertain. Of these, how many the detector flagged is shown next.")
    b.metric("Detector caught those", f"{result['detector_caught_when_fooled']} / {result['fooled_you']}")
    c.metric("Real voices you flagged", f"{result['reals_you_flagged']} / {result['reals']}")
    d.metric("Detector false alarms", f"{result['detector_false_alarms']} / {result['reals']}")
    st.caption(f"You caught {result['you_caught']} of {result['fakes']} fakes outright; the detector caught {result['detector_caught']} of {result['fakes']}. "
               f"Uncertain counts as not caught. Twenty clips is a small sample; see the modern-voice evaluation in Detector Lab for the full test.")
    table = pd.DataFrame(result["clips_detail"]).rename(columns={"clip": "Clip", "truth": "Truth", "generator": "Generator", "you": "You",
                                                               "confidence": "Your confidence", "detector_score": "Detector score", "detector": "Detector"})
    st.dataframe(table, hide_index=True, width="stretch")
    cue_table(cue_analysis([r for r in review_rows(db) if r["phase"] == "challenge" and r["role"] == "primary"]))


def cue_table(cues):
    st.markdown("#### Which cues held up")
    if not cues:
        st.caption("No perceptual cues were recorded on revealed clips.")
        return
    st.dataframe(pd.DataFrame([{"Cue": c["cue"], "Leans": c["leans"], "Times you marked it": c["marked"],
                                "Synthetic when marked": f"{c['synthetic']}/{c['marked']} ({c['synthetic_share']:.0%})"} for c in cues]),
                 hide_index=True, width="stretch")
    st.caption(f"Base rate: {cues[0]['base_rate']:.0%} of these clips were synthetic. A cue is informative when its share differs clearly "
               "from the base rate in the direction it leans. Counts this small are descriptive, not proof.")


def findings_screen(db):
    heading("Findings & Export", "TURN REVIEW INTO EVIDENCE", "See what your work establishes, then package it for another person to inspect.")
    if db.setting("reviewer_mode"):
        st.info("Return completed independent reviews from Quality Review.")
        return
    data = findings(db)
    calibration = calibration_feedback(db)
    state = progress(db, db.setting("primary_reviewer", "primary"))
    if data["kind"] == "challenge":
        challenge_panel(db, state)
    elif data["revealed_noncalibration"]:
        st.subheader("Main-study results")
        feedback_metrics(data["human"])
        a, b = st.columns(2)
        a.metric("Within-reviewer agreement", fmt(data["within_reviewer"]["agreement"]))
        b.metric("Independent agreement", fmt(data["independent"]["agreement"]))
        st.subheader("Where the published detector struggles")
        detector = data["detector"]
        a, b = st.columns(2)
        a.metric("Genuine speech rejected", f"{detector['genuine_rejected']} / {detector['genuine_total']}")
        b.metric("Spoofed speech accepted", f"{detector['spoof_accepted']} / {detector['spoof_total']}")
        cue_table(data["cues"])
        field = st.selectbox("Break down results by", ["codec", "attack", "guideline", "phase"])
        st.dataframe(pd.DataFrame(data["breakdowns"][field]), hide_index=True, width="stretch")
        st.caption("Published LFCC-GMM scores; no fresh inference. Calibration and repeats do not inflate main-study counts.")
    elif not calibration["human"]["reviewed"]:
        st.markdown('<div class="hero"><p class="eyebrow">YOUR EVIDENCE STARTS WITH CALIBRATION</p><h2>Get your first comparison.</h2><p>Complete the calibration batch, then reveal it in Quality Review. You will see where your listening judgments matched the references and which cases deserve a closer look.</p></div>', unsafe_allow_html=True)
        if st.button(state["next"]["label"], type="primary"):
            navigate(state["next"]["screen"], state["next"]["phase"])
    if calibration["human"]["reviewed"]:
        with st.container(border=True):
            st.markdown("#### Calibration feedback")
            st.caption("A learning checkpoint. Excluded from main-study performance.")
            feedback_metrics(calibration["human"])
            if st.button("Investigate calibration cases"):
                navigate("Quality Review", "calibration")
    st.subheader("Your research handoff")
    case_count = db.one("SELECT COUNT(*) AS n FROM cases")["n"]
    st.caption(f"{data['submitted_unique']} primary reviews submitted · {data['revealed_noncalibration']} main-study recordings revealed · {case_count} case studies written")
    st.download_button("Download research bundle", research_bundle(db), "audio_review_research.zip", "application/zip", type="primary",
                       help="Annotations, revision history, full case notes, adjudications, metrics, and a printable report. Revealed cases only.")
    if not case_count:
        st.caption("The report is still a draft. Investigate revealed cases and write findings before using it as portfolio evidence.")
    with st.expander("Individual files"):
        a, b = st.columns(2)
        a.download_button("Annotations · CSV", annotation_csv(db), "annotations.csv", "text/csv")
        b.download_button("Revision history · CSV", history_csv(db), "revision_history.csv", "text/csv")
        a.download_button("Findings report · HTML", report_html(db), "findings.html", "text/html")
        b.download_button("Metrics · JSON", json.dumps(data, indent=2), "metrics.json", "application/json")
        a.download_button("Case studies · CSV", cases_csv(db), "cases.csv", "text/csv")
        b.download_button("Adjudications · CSV", adjudications_csv(db), "adjudications.csv", "text/csv")
    with st.expander("What these numbers mean"):
        st.markdown("**Accuracy among decided** is shown with uncertainty, so abstaining is visible. **Agreement** compares original judgments, including uncertain. Undefined metrics are unavailable. Different guideline batches support descriptive comparisons. This benchmark does not establish real-world fraud-detection performance.")


def main():
    request = st.session_state.pop("nav_request", None)
    if request:
        screen, phase = request
        st.session_state["nav_main"] = screen
        if phase:
            st.session_state["batch_selector"] = phase
            st.session_state["quality_phase"] = phase
    db = store()
    try:
        reviewer = db.setting("primary_reviewer", "primary")
        state = progress(db, reviewer)
        with st.sidebar:
            st.markdown('<div class="brand"><span>◉</span>Audio Review Lab</div>', unsafe_allow_html=True)
            st.caption("Listen first. Label blind. Verify after.")
            workspace_switcher()
            st.divider()
            screen = st.radio("Workspace", ["Workspace", "Review", "Quality Review", "Findings & Export", "Detector Lab"], key="nav_main", label_visibility="collapsed")
            st.divider()
            st.caption("STUDY PROGRESS")
            if state["total"]:
                st.progress(state["done"] / state["total"], text=f"{state['done']} / {state['total']} reviewed")
            st.caption("REVIEWER")
            st.write(reviewer)
            with st.expander("Workspace details"):
                st.caption("Local workspace · audio study · original files preserved")
                st.download_button("Blind queue · CSV", csv_bytes(db.queue(reviewer), ["id", "phase", "role", "guideline", "due_at", "submitted_at"]), "blind_queue.csv", "text/csv")
        kind = db.setting("dataset_kind", "empty")
        if screen == "Detector Lab":
            st.markdown('<div class="workspace-badge">AFTER-REVIEW COMPARISON / LOCAL INFERENCE</div>', unsafe_allow_html=True)
        elif kind == "demo":
            st.warning("Practice workspace · generated tones and arbitrary test labels. Results are software demonstrations, not study evidence.")
        else:
            label = ("INDEPENDENT REVIEW" if db.setting("reviewer_mode") else
                     "LISTENING CHALLENGE / MODERN VOICES" if kind == "challenge" else "ASVSPOOF 2021 / PERSONAL STUDY")
            st.markdown(f'<div class="workspace-badge">{label}</div>', unsafe_allow_html=True)
        if "flash" in st.session_state:
            st.success(st.session_state.pop("flash"))
        if screen == "Detector Lab":
            forensics_screen(db)
        elif screen == "Workspace":
            overview_screen(db, reviewer)
        elif screen == "Review":
            review_screen(db, reviewer)
        elif screen == "Quality Review":
            quality_screen(db, reviewer)
        else:
            findings_screen(db)
    finally:
        db.close()


main()

