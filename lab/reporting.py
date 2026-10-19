"""Only explicitly revealed original submissions enter findings and exports."""
from collections import defaultdict
import csv
from html import escape
from io import StringIO
import json
from io import BytesIO
import zipfile

from .metrics import agreement, confusion, human_metrics
from .store import HUMAN_CUES, SYNTHETIC_CUES, canonical


def csv_bytes(rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        clean = {}
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, (dict, list)):
                value = canonical(value)
            # Excel/Sheets may execute formulas even inside quoted CSV cells.
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            clean[key] = value
        writer.writerow(clean)
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def review_rows(store):
    result = []
    config = store.one("SELECT config FROM threshold_config")
    config = json.loads(config["config"]) if config else {}
    threshold = config.get("threshold")
    # LFCC-GMM scores support human speech when high; challenge detector scores support synthetic.
    high = "synthetic" if config.get("direction") == "higher_is_synthetic" else "human"
    low = "human" if high == "synthetic" else "synthetic"
    for r in store.revealed_rows():
        payload = json.loads(r.pop("payload"))
        r.update(payload)
        r["prediction"] = None if r["score"] is None or threshold is None else (high if r["score"] >= threshold else low)
        result.append(r)
    return result


def cue_analysis(rows):
    """How often each perceptual cue coincided with a synthetic reference. Descriptive, not causal."""
    rows = [r for r in rows if r.get("reference") in ("human", "synthetic")]
    base = sum(r["reference"] == "synthetic" for r in rows) / len(rows) if rows else None
    result = []
    for cue in sorted({c for r in rows for c in r.get("cues", [])}):
        marked = [r for r in rows if cue in r.get("cues", [])]
        synthetic = sum(r["reference"] == "synthetic" for r in marked)
        result.append({"cue": cue, "leans": "synthetic" if cue in SYNTHETIC_CUES else "human" if cue in HUMAN_CUES else "—",
                       "marked": len(marked), "synthetic": synthetic, "human": len(marked) - synthetic,
                       "synthetic_share": synthetic / len(marked), "base_rate": base})
    return sorted(result, key=lambda r: (-r["marked"], r["cue"]))


def challenge_summary(rows):
    """You vs. the detector on revealed challenge clips. 'Fooled' = a fake you judged human or uncertain."""
    rows = [r for r in rows if r["phase"] == "challenge" and r["role"] == "primary"]
    fakes = [r for r in rows if r["reference"] == "synthetic"]
    reals = [r for r in rows if r["reference"] == "human"]
    fooled = [r for r in fakes if r["judgment"] != "synthetic"]
    return {"clips": len(rows), "fakes": len(fakes), "reals": len(reals),
            "you_caught": len(fakes) - len(fooled), "fooled_you": len(fooled),
            "fooled_judged_human": sum(r["judgment"] == "human" for r in fooled),
            "detector_caught": sum(r["prediction"] == "synthetic" for r in fakes),
            "detector_caught_when_fooled": sum(r["prediction"] == "synthetic" for r in fooled),
            "reals_you_flagged": sum(r["judgment"] == "synthetic" for r in reals),
            "reals_you_uncertain": sum(r["judgment"] == "uncertain" for r in reals),
            "detector_false_alarms": sum(r["prediction"] == "synthetic" for r in reals),
            "clips_detail": [{"clip": r["trial_id"], "truth": r["reference"], "generator": r["attack"],
                              "you": r["judgment"], "confidence": r["confidence"], "detector_score": r["score"],
                              "detector": r["prediction"]} for r in rows]}


def findings(store):
    all_rows = review_rows(store)
    primary = [r for r in all_rows if r["role"] == "primary" and r["phase"] != "calibration"]
    scored = [r for r in primary if r["prediction"] is not None]
    pairs = {"repeat": ([], []), "independent": ([], [])}
    primary_by_recording = {r["recording_id"]: r for r in primary}
    for r in all_rows:
        if r["role"] not in pairs or r["recording_id"] not in primary_by_recording:
            continue
        original = primary_by_recording[r["recording_id"]]
        # Agreement by guideline version is shown separately in detailed exports.
        if original["guideline"] == r["guideline"]:
            pairs[r["role"]][0].append(original["judgment"])
            pairs[r["role"]][1].append(r["judgment"])
    breakdown = {}
    for field in ("codec", "attack", "guideline", "phase"):
        groups = defaultdict(list)
        for r in primary:
            groups[r[field]].append(r)
        breakdown[field] = [{field: name, **human_metrics(items),
                             **confusion([r for r in items if r["prediction"] is not None])}
                            for name, items in sorted(groups.items())]
    primary_reviewer = store.setting("primary_reviewer", "primary")
    counts = {r["phase"]: r["n"] for r in store.rows("SELECT phase,COUNT(*) AS n FROM assignments WHERE role='primary' AND submitted_at IS NOT NULL GROUP BY phase")}
    repeat = agreement(*pairs["repeat"])
    independent = agreement(*pairs["independent"])
    return {"kind": store.setting("dataset_kind", "empty"), "reviewer": primary_reviewer,
            "submitted_unique": sum(counts.values()), "phase_submissions": counts,
            "milestone_100": store.setting("dataset_kind") == "research" and counts.get("calibration", 0) >= 20 and counts.get("main-01", 0) >= 80,
            "revealed_noncalibration": len(primary), "human": human_metrics(primary),
            "detector": confusion(scored), "missing_scores": len(primary) - len(scored),
            "within_reviewer": repeat, "independent": independent, "breakdowns": breakdown, "cues": cue_analysis(primary)}


def calibration_feedback(store):
    rows = [r for r in review_rows(store) if r["phase"] == "calibration" and r["role"] == "primary"]
    return {"human": human_metrics(rows), "detector": confusion([r for r in rows if r["prediction"] is not None]),
            "purpose": "Calibration feedback only; excluded from main-study performance."}


def research_bundle(store):
    stream = BytesIO()
    files = {"annotations.csv": annotation_csv(store), "revision_history.csv": history_csv(store),
             "cases.csv": cases_csv(store), "adjudications.csv": adjudications_csv(store),
             "metrics.json": json.dumps(findings(store), indent=2), "findings.html": report_html(store),
             "study_manifest.json": json.dumps(manifest_export(store), indent=2),
             "README.txt": "Audio Review Lab research handoff. Original submitted judgments and revealed cases only. Calibration is excluded from study metrics. Report limitations and incomplete work honestly. Audio is not included. See findings.html and the companion CSV records."}
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return stream.getvalue()


def annotation_csv(store):
    fields = ["assignment_id", "trial_id", "reviewer", "phase", "role", "guideline", "submitted_at",
              "intelligibility", "tags", "cues", "judgment", "confidence", "observations", "rationale", "escalated", "escalation_reason",
              "reference", "score", "prediction", "codec", "attack", "kind"]
    return csv_bytes(review_rows(store), fields)


def history_csv(store):
    rows = store.rows("""SELECT e.id,e.assignment_id,a.reviewer,a.phase,a.guideline,e.event,e.payload,e.created_at
                       FROM annotation_revisions e JOIN assignments a ON a.id=e.assignment_id
                       WHERE a.revealed_at IS NOT NULL ORDER BY e.id""")
    return csv_bytes(rows, ["id", "assignment_id", "reviewer", "phase", "guideline", "event", "payload", "created_at"])


def cases_csv(store):
    return csv_bytes(store.rows("SELECT r.trial_id,c.title,c.observation,c.hypothesis,c.follow_up,c.created_at FROM cases c JOIN recordings r ON r.id=c.recording_id ORDER BY c.id"),
                     ["trial_id", "title", "observation", "hypothesis", "follow_up", "created_at"])


def adjudications_csv(store):
    return csv_bytes(store.rows("SELECT r.trial_id,a.reviewer,a.decision,a.rationale,a.created_at FROM adjudications a JOIN recordings r ON r.id=a.recording_id ORDER BY a.id"),
                     ["trial_id", "reviewer", "decision", "rationale", "created_at"])


def manifest_export(store):
    """Full private manifest only after every primary assignment has been revealed."""
    manifest = store.study_manifest()
    row = store.one("SELECT sha256 FROM study")
    if not row:
        return {}
    pending = store.one("SELECT 1 FROM assignments WHERE role='primary' AND revealed_at IS NULL")
    if pending:
        return {"manifest_sha256": row["sha256"], "seed": manifest["seed"], "counts": manifest["counts"],
                "selection": "Withheld until all primary batches are revealed", "import_provenance": manifest["import_provenance"]}
    return {**manifest, "manifest_sha256": row["sha256"]}


def report_html(store):
    data = findings(store)
    cases = store.rows("SELECT c.*,r.trial_id FROM cases c JOIN recordings r ON r.id=c.recording_id ORDER BY c.id LIMIT 5")
    config = store.one("SELECT config FROM threshold_config")
    config = json.loads(config["config"]) if config else {}
    provenance = store.one("SELECT content FROM provenance WHERE id=?", (store.setting("import_provenance"),))
    provenance = json.loads(provenance["content"]) if provenance else {}
    manifest = store.one("SELECT sha256 FROM study")
    def value(v):
        if v is None:
            return "not measured"
        return f"{v:.1%}" if isinstance(v, float) else str(v)
    def table(items):
        return "<table>" + "".join(f"<tr><th>{escape(k)}</th><td>{escape(value(v))}</td></tr>" for k, v in items.items()) + "</table>"
    def kappa(metric):
        if not metric["pairs"]:
            return "not measured"
        return "unavailable (constant categories)" if metric["kappa"] is None else f"{metric['kappa']:.3f}"
    def brief(text, limit=160):
        return escape(text if len(text) <= limit else text[:limit-1] + "…")
    case_html = "".join(f"<article><h3>{brief(c['title'], 70)} · {escape(c['trial_id'])}</h3><p><b>Observed:</b> {brief(c['observation'])}<br><b>Possible explanation:</b> {brief(c['hypothesis'], 120)}<br><b>Follow-up:</b> {brief(c['follow_up'], 120)}</p></article>" for c in cases)
    label, scope, detector_title = {
        "demo": ("DEMONSTRATION — generated tones; arbitrary fixture labels; no hiring evidence",
                 "Software demonstration with generated non-speech tones and arbitrary fixture scores. No actual detector output or speech-study results.",
                 "Fixture scores — software checks only"),
        "challenge": ("LISTENING CHALLENGE — original blind judgments vs. a frozen detector; revealed clips only",
                      "Blind listening test on fresh modern voice-cloning output and real LibriSpeech speech, compared with the lab's detector, whose scores were frozen before listening.",
                      "Lab detector (fresh local inference)"),
    }.get(data["kind"], ("PERSONAL STUDY — original submitted judgments; revealed cases only",
                         "ASVspoof 2021 LA benchmark analysis using published LFCC-GMM scores; no model training or new inference.",
                         "Published detector results"))
    challenge = data["kind"] == "challenge"
    cue_html = ("<table><tr><th>Cue</th><th>Leans</th><th>Marked</th><th>Synthetic when marked</th></tr>"
                + "".join(f"<tr><td>{escape(c['cue'])}</td><td>{c['leans']}</td><td>{c['marked']}</td><td>{c['synthetic']}/{c['marked']} ({c['synthetic_share']:.0%})</td></tr>" for c in data["cues"])
                + f"</table><p>Base rate: {value(data['cues'][0]['base_rate'])} of revealed clips were synthetic. Descriptive only; small counts are unreliable.</p>"
                if data["cues"] else "<p>No perceptual cues recorded on revealed clips yet.</p>")
    threshold_note = (f"Detector threshold: {escape(str(config.get('threshold', 'not frozen')))}; higher scores indicate synthetic speech. Frozen on a separate calibration split before the challenge was prepared. Receipt fingerprint: {escape(str(config.get('receipt_sha256', 'not available')))}."
                      if challenge else
                      f"Threshold: {escape(str(config.get('threshold', 'not frozen')))}; progress calibration n={config.get('n', 0)}; higher scores support human speech. Empirical equal-error operating point, frozen before sampling. Import fingerprint: {escape(str(config.get('import_provenance', 'not available')))}.")
    source_note = ("Clips are matched by speaker and text: each fake clones a LibriSpeech speaker reading a sentence that speaker also recorded. Challenge copies are level-matched 16 kHz renderings so sample rate and loudness do not reveal origin. Twenty clips is a small sample."
                   if challenge else
                   "Source mapping groups human and voice-converted audio by VCTK source, and TTS by normalized input text. Cross-type identity of spoken text is not fully observable. Unmapped items and source overlap with progress calibration are excluded.")
    sampling_note = ("Fake clips passed a Whisper transcription check (WER ≤ 0.2) before any detector scoring; garbled generations were excluded and recorded in the frozen manifest."
                     if challenge else
                     "Sampling is conditioned on readable audio. Strict decoding and PCM-checksum failures are excluded and retained in the private manifest; this compatibility filter may affect which conditions are represented.")
    dataset_note = (f'Real speech: <a href="https://www.openslr.org/12">LibriSpeech</a>, CC BY 4.0. Fakes: open voice-cloning models generated locally (see the study manifest). Modern-set fingerprint: {escape(str(provenance.get("manifest_sha256", "not available")))}.'
                    if challenge else
                    f'Dataset: <a href="https://zenodo.org/records/4837263">ASVspoof 2021 Logical Access</a>; ODC-By attribution, official notices retained. Scores: <a href="https://github.com/asvspoof-challenge/2021">ASVspoof organizers, LFCC-GMM baseline</a>. Metadata SHA-256: {escape(provenance.get("metadata_sha256", "demo or unavailable"))}.')
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Audio Review Lab · Findings</title>
<style>body{{font:12px/1.4 system-ui;color:#152b3a;max-width:800px;margin:30px auto}}h1{{font-size:27px}}h2{{font-size:18px}}h3{{font-size:13px}}.badge{{padding:10px;background:#e5f1ed}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:4px}}article{{margin-bottom:14px}}p{{overflow-wrap:anywhere}}@media print{{@page{{size:A4;margin:16mm}}body{{margin:0;font-size:10px;line-height:1.35}}h1{{font-size:24px;margin:10px 0}}h2{{font-size:16px;margin:12px 0 6px}}h3{{font-size:11px;margin:0 0 4px}}p{{margin:0 0 8px}}.page{{break-after:page}}article{{break-inside:avoid;margin-bottom:10px}}article p{{margin:0}}th,td{{padding:3px}}}}</style>
<section class="page"><p>AUDIO REVIEW LAB / RESEARCH HANDOFF</p><h1>Speech annotation & quality review</h1><p class="badge">{escape(label)}</p>
<p>Scope: {scope} Completed submissions: {data['submitted_unique']}; revealed non-calibration cases: {data['revealed_noncalibration']}.</p>
<h2>Human review</h2>{table(data['human'])}<h2>{detector_title}</h2>{table(data['detector'])}
<h2>Perceptual cues</h2>{cue_html}
<h2>Quality assurance</h2><p>Within-reviewer: {escape(str(data['within_reviewer']['pairs']))} pairs; agreement {value(data['within_reviewer']['agreement'])}; kappa {kappa(data['within_reviewer'])}. Independent: {data['independent']['pairs']} pairs; agreement {value(data['independent']['agreement'])}; kappa {kappa(data['independent'])}.</p>
<p>{threshold_note}</p>
</section><section><h2>Investigated cases ({len(cases)}/5)</h2>{case_html or '<p>No case studies recorded. Listening observations and conclusions must be written by the reviewer.</p>'}
<h2>Limitations and recommended next work</h2><p>Balanced, source-deduplicated benchmark sample; results do not establish real-world fraud detection, authentication performance, or population prevalence. Perceived synthesis is not proof of fraud. Confidence is self-reported. No causal claim about guideline changes; batches differ. Agreement concerns original judgments before adjudication. Repeated recordings are not independent samples. Video was not evaluated.</p>
<p>{source_note}</p>
<p>{sampling_note}</p>
<p>Next work: complete outstanding reviews, resolve documented escalations, and obtain independent review where absent. Case summaries may be abbreviated; cases.csv retains complete notes. Supporting CSVs contain all observations; summaries never replace the evidence.</p>
<p>{dataset_note} Manifest SHA-256: {escape(manifest['sha256'] if manifest else 'not available')}.</p></section></html>"""
