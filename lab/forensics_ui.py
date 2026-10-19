"""Technical instrument panel, independent of the blind annotation workflow."""
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
import streamlit as st

from .audio import wav_bytes
from .detectors import DETECTORS, LABELS, installed, load, setup_command
from .forensics import RATE, analyze, explain, fmt_p, read_audio, report_html, save_report
from .learning import forensic_examples

ROOT = Path(__file__).resolve().parents[1]
MARKS = {"pass": "✓ Pass", "fail": "✗ Fail", "n/a": "— n/a"}
COMPONENT_LABELS = {"pellav2": "pellav2", "antideepfake": "AntiDeepfake"}
CHANNEL_LABELS = {"clean": "Clean", "mp3": "MP3 64 kbps", "phone": "Telephone 8 kHz", "noise20": "Noise 20 dB SNR"}


@st.cache_resource(show_spinner=False)
def detector(kind="pellav2"):
    return load(kind)


def why_panel(report):
    why = explain(report)
    with st.container(border=True):
        st.markdown("#### Why this result")
        st.markdown(f"**{why['verdict']}**")
        if why["components"]:
            cols = st.columns(len(why["components"]))
            for col, (name, value) in zip(cols, why["components"].items()):
                col.metric(f"{COMPONENT_LABELS.get(name, name)} alone", fmt_p(value),
                           help="Mean window score of this model before fusion. The fused score above is what decides.")
        st.dataframe(pd.DataFrame([{"Check": c["check"], "Measured": c["measured"], "Limit": c["limit"], "Result": MARKS[c["status"]]}
                                   for c in why["checks"]]), hide_index=True, width="stretch")
        st.caption("The vote compares the aggregate model score with the detector's threshold. The checks are fixed reliability rules; "
                   "any failure except the perturbation check makes the assessment inconclusive. None of them proves origin.")


def chart_style(fig, height=270):
    fig.update_layout(height=height, margin=dict(l=60, r=24, t=20, b=48),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f0f4f9",
                      font=dict(color="#52677f", size=11), showlegend=False)
    fig.update_xaxes(automargin=True, title_standoff=10)
    fig.update_yaxes(automargin=True, title_standoff=10)
    return fig


def signal_plots(x):
    # Extrema envelope preserves transient peaks at large zoom levels.
    stride = max(1, int(np.ceil(len(x)/2000)))
    trimmed = x[:len(x)//stride*stride].reshape(-1, stride)
    t = np.arange(len(trimmed))*stride/RATE
    fig = go.Figure()
    fig.add_scatter(x=t, y=trimmed.max(axis=1), mode="lines", line=dict(color="#3a9cbb", width=1))
    fig.add_scatter(x=t, y=trimmed.min(axis=1), mode="lines", fill="tonexty", line=dict(color="#3a9cbb", width=1))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Amplitude (full scale)")
    st.plotly_chart(chart_style(fig, 210), use_container_width=True, theme=None, config={"displayModeBar": False})
    f, t, z = signal.stft(x, RATE, window="hann", nperseg=400, noverlap=240, nfft=512, boundary=None)
    # Normalize to one common file maximum; this is a visualization, not dBFS power.
    magnitude = np.abs(z)
    peak = float(magnitude.max())
    db = 20*np.log10(np.maximum(magnitude, peak*1e-4)/peak) if peak else np.full(magnitude.shape, -80.)
    # Max-pool columns so short high-energy events survive display reduction.
    factor = max(1, int(np.ceil(len(t)/1200)))
    usable = len(t)//factor*factor
    display = db[:, :usable].reshape(len(f), -1, factor).max(axis=2)
    times = t[:usable].reshape(-1, factor).mean(axis=1)
    spec = go.Figure(go.Heatmap(x=times, y=f, z=display, zmin=-80, zmax=0, colorscale="Magma",
                              colorbar=dict(title="dB rel. peak", thickness=10)))
    spec.update_xaxes(title="Time (s)")
    spec.update_yaxes(title="Frequency (Hz)")
    st.plotly_chart(chart_style(spec, 330), use_container_width=True, theme=None)
    st.caption("STFT: Hann 25 ms · hop 10 ms · FFT 512 · 16 kHz mono. Color = magnitude dB relative to the file maximum. Display columns are max-pooled; no detector attribution is implied.")
    f, p = signal.welch(x, RATE, nperseg=512, noverlap=256)
    spectrum = go.Figure(go.Scatter(x=f, y=10*np.log10(np.maximum(p, 1e-20)), line=dict(color="#38649f")))
    spectrum.update_xaxes(title="Frequency (Hz)")
    spectrum.update_yaxes(title="PSD (dBFS²/Hz)")
    st.plotly_chart(chart_style(spectrum, 240), use_container_width=True, theme=None)
    st.caption("Welch power spectral density: Hann 512 samples, 50% overlap. Compression and noise alter spectra in both genuine and synthetic speech.")


def render_report(report, x):
    assessment = report["assessment"]
    score = assessment["synthetic_score"]
    title = assessment["assessment"]
    if title == "Inconclusive":
        title = assessment["binary_model_decision"].title() + " vote · inconclusive"
    st.markdown(f'<div class="hero"><p class="eyebrow">MODEL ASSESSMENT / FRESH LOCAL INFERENCE</p><h2>{escape(title)}</h2><p>{escape(report["model"].get("name", "Wav2Vec2-XLSR"))} · {escape(report["model"].get("description", "Local classifier"))}</p></div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Synthetic-class score", f"{score:.3f}")
    b.metric("Windows analyzed", len(report["windows"]))
    c.metric("Inference + analysis", f"{report['runtime_seconds']:.1f} s")
    policy = report["policy"]
    threshold, low, high = policy["binary_threshold"], policy["human_below"], policy["synthetic_above"]
    st.caption(f"Speech presence and language are not verified. The score is a model output, not a measured probability of AI origin. The abstention band {low:.2f}–{high:.2f} is an engineering rule, not a validated confidence interval.")
    why_panel(report)
    for reason in assessment["abstention_reasons"]:
        st.warning(reason)
    for warning in report["warnings"]:
        st.warning(warning)
    evidence, acoustics, robustness, provenance = st.tabs(["Detector evidence", "Signal laboratory", "Sensitivity tests", "Method & provenance"])
    with evidence:
        st.subheader("Where the detector changes its assessment")
        scores = report["windows"]
        fig = go.Figure()
        for row in scores:
            fig.add_trace(go.Scatter(x=[row["start"], row["end"]], y=[row["synthetic_score"]]*2,
                                    mode="lines", line=dict(width=5, color="#bb593e"),
                                    hovertemplate="%{x:.2f} s<br>Synthetic score %{y:.3f}<extra></extra>"))
        fig.add_hrect(y0=low, y1=high, fillcolor="#b6c2d2", opacity=.25, line_width=0)
        fig.add_hline(y=threshold, line_dash="dot", line_color="#8292a6")
        fig.update_xaxes(title="Time (s)", range=[0, report["source"]["duration_seconds"]])
        fig.update_yaxes(title="Synthetic-class score", range=[0, 1])
        st.plotly_chart(chart_style(fig, 280), use_container_width=True, theme=None)
        rows = [{"Start (s)": r["start"], "End (s)": r["end"], "Synthetic score": r["synthetic_score"],
                 **{f"{COMPONENT_LABELS.get(k, k)} score": v for k, v in r.get("components", {}).items()},
                 "Human score": r["human_score"], "Human logit": r["logits"][0], "Synthetic logit": r["logits"][1]} for r in scores]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Window length: {report['preprocessing'].get('window_samples', 96000)/RATE:.4f} s. {report['preprocessing']['window_tail']}. {assessment['aggregation']}. Window scores are not edit-boundary predictions.")
        index = st.selectbox("Replay an analyzed window", range(len(scores)), format_func=lambda i: f"{scores[i]['start']:.2f}–{scores[i]['end']:.2f} s · score {scores[i]['synthetic_score']:.3f}")
        segment = scores[index]
        st.audio(wav_bytes(x, RATE, segment["start"], segment["end"]), format="audio/wav")
    with acoustics:
        st.subheader("Measurements, not visual guesswork")
        signal_plots(x)
        m = report["diagnostics"]
        a, b, c, d = st.columns(4)
        a.metric("RMS level", f"{m['rms_dbfs']:.1f} dBFS")
        b.metric("Clipped samples", f"{m['clipped_sample_fraction']:.2%}")
        c.metric("Energy activity", f"{m['active_frame_fraction']:.1%}")
        d.metric("Spectral centroid", f"{m['spectral_centroid_hz']:.0f} Hz")
        st.json(m, expanded=True)
        st.caption("Energy activity is not voice activity detection. Spectral flatness, bandwidth, clipping, and zero crossings describe the signal; none establishes synthetic origin or explains the transformer's decision.")
    with robustness:
        st.subheader("Does the result survive small audio changes?")
        if report["robustness"]:
            st.dataframe(pd.DataFrame(report["robustness"]), hide_index=True, width="stretch")
            delta = max(abs(r["score_change"]) for r in report["robustness"])
            st.metric("Largest absolute score change", f"{delta:.3f}")
            st.caption("Three fresh inferences on the same most energetic window: half gain, 3.4 kHz low-pass, and seeded 30 dB SNR Gaussian noise. Each is compared with that window's unmodified score. Stability does not establish correctness.")
        else:
            st.info("Enable sensitivity tests before analysis to run three controlled perturbations.")
    with provenance:
        model = report["model"]
        st.markdown(f"**Checkpoint:** [{model['repository']}]({report['sources'][0]})")
        st.code(model['revision'], language=None)
        st.write(f"{model['parameters']:,} parameters · CPU float32 · pinned checksums")
        if model.get('training_domain'):
            st.markdown(f"**Training domain:** {model['training_domain']}")
        st.markdown("**Class mapping:** " + " · ".join(f"logit {k} = {v}" for k, v in model.get('class_mapping', {}).items()))
        st.markdown("**Interpretation limits**")
        for item in report["limitations"]:
            st.write("• " + item)
        with st.expander("Exact preprocessing, policy, checksums, and runtime"):
            st.json({k: report[k] for k in ("preprocessing", "policy", "source", "model", "environment")})
    st.divider()
    a, b = st.columns(2)
    a.download_button("Export reproducible analysis · JSON", json.dumps(report, indent=2, allow_nan=False), "audio_forensics.json", "application/json")
    b.download_button("Export technical report · HTML", report_html(report), "audio_forensics.html", "text/html")
    st.caption("Analysis JSON is saved locally in data/analyses. Uploaded audio stays in this local app session and is not saved to the study database or sent to a hosted inference API.")


def rate(cell):
    low, high = cell['wilson95'] or (0, 0)
    return f"{cell['flagged_synthetic']} / {cell['n']} ({cell['rate']:.1%}; 95% CI {low:.1%}–{high:.1%})"


def modern_evaluation(evaluation):
    test, bar = evaluation['test'], evaluation['bar']
    with st.expander("Measured performance · modern voices", expanded=False):
        st.caption(f"Fresh clips from modern open voice-cloning models and matched real LibriSpeech speech, clean condition. "
                   f"Threshold {evaluation['threshold']:.2f} frozen on a separate calibration split before the test was scored once.")
        a, b = st.columns(2)
        a.metric('Modern fakes flagged synthetic', f"{test['fakes']['flagged_synthetic']} / {test['fakes']['n']}")
        b.metric('Real voices wrongly flagged', f"{test['reals']['flagged_synthetic']} / {test['reals']['n']}")
        verdict = (f"Pre-set bar: ≥{bar['min_fake_detection']:.0%} of fakes flagged and ≤{bar['max_real_false_alarms']:.0%} of real speech flagged. "
                   + ("Met." if evaluation['bar_met'] else "Not met."))
        (st.success if evaluation['bar_met'] else st.warning)(verdict)
        rows = [{'Clip set': name, 'Flagged synthetic': rate(cell)} for name, cell in
                [*((f'Fake · {g}', c) for g, c in test['by_generator'].items()), *((f'Real · {s}', c) for s, c in test['by_source'].items())]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.markdown("**Channel conditions** (same clips, re-encoded)")
        st.dataframe(pd.DataFrame([{'Channel': name, 'Fakes caught': rate(c['fakes']), 'Real false alarms': rate(c['reals'])}
                                   for name, c in test['by_channel'].items()]), hide_index=True, width='stretch')
        if evaluation.get('comparison'):
            st.markdown("**Older detectors on the same clean test clips**")
            st.dataframe(pd.DataFrame([{'Model': LABELS.get(k, k), 'Fakes caught': rate(c['fakes']), 'Real false alarms': rate(c['reals'])}
                                       for k, c in evaluation['comparison'].items()]), hide_index=True, width='stretch')
        legacy = ROOT / 'docs/detector-evaluation-v2.json'
        if legacy.exists():
            old = json.loads(legacy.read_text())['test']['summary']
            st.caption(f"Earlier ASVspoof 2019-era protocol (v2): AASIST {old['correct']}/{old['n']} correct. Those attacks predate modern voice cloning.")
        for note in evaluation['limitations']:
            st.caption(note)
        st.download_button('Download full evaluation record', json.dumps(evaluation, indent=2), 'detector_evaluation.json', 'application/json')


def fusion_evaluation(evaluation):
    test = evaluation['test']
    with st.expander("Measured performance · modern voices, four channels", expanded=False):
        st.caption(f"Fresh speakers and fresh clips, scored once. Fusion weights and threshold "
                   f"({fmt_p(evaluation['threshold'])}) were fixed on separate development data first. "
                   f"{evaluation['heldout_generator'].upper()} never appeared in development.")
        a, b = st.columns(2)
        a.metric('Modern fakes flagged (clean)', f"{test['fakes']['flagged_synthetic']} / {test['fakes']['n']}")
        b.metric('Real voices wrongly flagged (clean)', f"{test['reals']['flagged_synthetic']} / {test['reals']['n']}")
        (st.success if evaluation['bar_met'] else st.warning)(
            f"Pre-set bar: {evaluation['bar']['summary']}. " + ("Met." if evaluation['bar_met'] else
            "Not met: " + '; '.join(evaluation['bar_failures']) + '.'))
        rows = []
        for ch, cell in test['by_channel'].items():
            rows.append({'Channel': CHANNEL_LABELS.get(ch, ch),
                         **{f'Fakes · {g}': rate(c) for g, c in cell['by_generator'].items()},
                         'Real false alarms': rate(cell['reals'])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.markdown("**Each model alone on the same clips**")
        st.dataframe(pd.DataFrame([{'Model': name, 'Channel': CHANNEL_LABELS.get(ch, ch), 'Fakes caught': rate(c['fakes']),
                                    'Real false alarms': rate(c['reals'])}
                                   for name, result in evaluation['comparison'].items() for ch, c in result['by_channel'].items()]),
                     hide_index=True, width='stretch')
        for note in evaluation['limitations']:
            st.caption(note)
        st.download_button('Download full evaluation record', json.dumps(evaluation, indent=2), 'detector_evaluation.json', 'application/json')


def legacy_evaluation(evaluation):
    with st.expander("Legacy ASVspoof 2019-era evaluation", expanded=False):
        st.caption('Both models were compared on 40 source-separated recordings. The choice was frozen before scoring 40 fresh test recordings. No test threshold fitting. These are 2019-era attacks, not modern voice cloning.')
        summary = evaluation['test']['summary']
        low, high = summary['accuracy_wilson95']
        st.metric('Selected detector: raw test predictions correct', f"{summary['correct']} / {summary['n']}")
        st.caption(f"95% Wilson interval: {low:.1%}–{high:.1%}. {summary['false_positive']} human false positives / {summary['human']} human; {summary['false_negative']} synthetic misses / {summary['synthetic']} synthetic.")
        st.write(f"Reliability policy: {summary['abstained']} abstentions; {summary['correct_among_decided']} correct / {summary['decided']} decided cases. Binary accuracy and policy-qualified accuracy measure different things.")
        st.dataframe(pd.DataFrame([{'Model':k, 'Selection correct':v['summary']['correct'], 'Selection n':v['summary']['n'], 'Human false positives':v['summary']['false_positive']} for k,v in evaluation['selection'].items()]), hide_index=True, width='stretch')
        for note in evaluation['limitations']:
            st.caption(note)
        st.download_button('Download full evaluation record', json.dumps(evaluation,indent=2), 'detector_evaluation.json', 'application/json')


def forensics_screen(db):
    st.markdown('<p class="eyebrow">CHECK YOUR EARS AGAINST A MODEL · NEVER BEFORE</p>', unsafe_allow_html=True)
    st.title("Detector Lab")
    st.caption("A speech deepfake detector for comparison after blind review. Inspect its evidence and challenge its result.")
    blind = db.one("SELECT COUNT(*) AS n FROM assignments WHERE revealed_at IS NULL")["n"]
    if blind:
        st.info(f"{blind} clips in this workspace are still blind. Judge them in Review before using the detector, so its "
                "output cannot influence your labels. This lab never shows study clips; it only scores audio you upload.")
    evaluation_path = ROOT / 'docs/detector-evaluation.json'
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    modern = bool(evaluation) and evaluation.get('version') in ('3.0', '4.0')
    recommended = evaluation['selected_model'] if modern else ('pellav2' if installed('pellav2') else 'aasist')
    kinds = list(DETECTORS)
    kind = st.selectbox('Detector', kinds, index=kinds.index(recommended),
                        format_func=lambda k: LABELS[k] + (' · selected by modern-voice evaluation' if modern and k == recommended else ''))
    available = installed(kind)
    a, b, c = st.columns(3)
    a.metric("Inference", "Local / CPU")
    b.metric("Model", LABELS[kind])
    c.metric("Checkpoint", "Ready" if available else "Not installed")
    with st.expander("Scope: what this instrument can determine"):
        st.write("This instrument classifies speech using learned human/synthetic acoustic patterns. It can give a reproducible model assessment, temporal scores, signal measurements, and sensitivity evidence. It cannot prove recording authenticity. Unseen generators, languages, codecs, music, and mixed sources can produce confident errors. It has not been validated on your intended real-world distribution.")
    if not available:
        st.info("Install the pinned detector once; later inference runs offline.")
        st.code(setup_command(kind), language="bash")
    if modern and evaluation['version'] == '4.0':
        fusion_evaluation(evaluation)
    elif modern:
        modern_evaluation(evaluation)
    elif evaluation:
        legacy_evaluation(evaluation)
    lessons = forensic_examples(db) if not db.setting("reviewer_mode") else []
    source_options = ["Upload audio"] + (["Reference demonstration"] if lessons else [])
    choice = st.radio("Input source", source_options, index=1 if lessons else 0, horizontal=True)
    blob = None
    if choice == "Upload audio":
        upload = st.file_uploader("Drop a speech recording", type=["wav", "flac", "mp3", "ogg", "aiff", "aif"],
                                  help="0.5–120 seconds · up to 50 MiB · 1–8 channels. Inference and preprocessing run locally.")
        if upload is not None:
            blob = upload.getvalue()
    else:
        index = st.selectbox("Reference recording", range(len(lessons)), format_func=lambda i: f"Reference {i+1} · {lessons[i]['label'].title()} · {lessons[i]['duration']:.2f} s")
        row = db.one("SELECT path FROM recordings WHERE id=?", (lessons[index]["id"],))
        blob = Path(row["path"]).read_bytes()
        st.info(f"Known dataset origin: {lessons[index]['label'].upper()}. The detector prediction below is a separate measurement.")
        st.caption("First duration-qualified examples in the frozen evaluation order; selected before scoring. Outside your annotation study. No selection for correct or confident predictions.")
    if blob is None:
        st.markdown('<div class="hero"><p class="eyebrow">START AN INVESTIGATION</p><h2>Bring a recording. Get measurable evidence.</h2><p>Learned speech classification, window-level scores, spectral analysis, controlled perturbations, and a technical report with exact model provenance.</p></div>', unsafe_allow_html=True)
        return
    fingerprint = sha256(blob).hexdigest()
    try:
        audio, rate, source = read_audio(blob)
    except ValueError as error:
        st.error(str(error))
        return
    st.audio(wav_bytes(audio, rate), format="audio/wav")
    st.caption(f"{source['duration_seconds']:.2f} s · {rate:,} Hz · {source['channels']} channel(s) · {source['format']}/{source['subtype']} · SHA-256 {fingerprint[:16]}…")
    if source["duration_seconds"] < 2.5:
        st.warning(f"This clip is {source['duration_seconds']:.2f} s; the lab requires 2.50 s for a qualified assessment. Inference can run, but the reliability policy will abstain.")
    sensitivity = st.checkbox("Run sensitivity tests", value=True, help="Three additional model runs on the same most energetic window. Slower but exposes score fragility.")
    if st.button("Analyze recording", type="primary", disabled=not available):
        try:
            with st.spinner("Loading verified checkpoint and running local inference…"):
                model = detector(kind)
                bar = st.progress(0, text="Preparing audio")
                result, x = analyze(blob, model, robustness=sensitivity, on_progress=lambda value, text: bar.progress(value, text=text))
                save_report(result)
                bar.empty()
            st.session_state["forensic_result"] = (fingerprint, sensitivity, result, x, kind)
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            st.error(f"Analysis could not complete: {error}")
    saved = st.session_state.get("forensic_result")
    if saved and len(saved) == 5 and saved[0] == fingerprint and saved[1] == sensitivity and saved[4] == kind:
        render_report(saved[2], saved[3])
