"""Build the fixed local annotation demonstration without model inference."""
from pathlib import Path
from hashlib import sha256
from io import StringIO
import csv
import html
import json
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.source_measurements import measure_source

EXPECTED = '299c3e4a38abd3e74c61b39ec52824f406f75c62673b0220db5167b3aaabb640'
DIRECTORY = ROOT/'studio/dist/samples'


def level(x):
    value = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return float(20*np.log10(value)) if value else None


def annotations(audio, rate):
    """Cover every sample once with measurements; flag quiet spans separately."""
    rows = []
    for start in range(0, len(audio), 5*rate):
        end = min(start+5*rate, len(audio))
        x = audio[start:end]
        peak = float(np.max(np.abs(x)))
        rms = level(x)
        near = int(np.count_nonzero(np.abs(x) >= .999))
        rows.append({'start': start/rate, 'end': end/rate, 'kind': 'interval_review',
                     'observation': f'RMS {rms:.2f} dBFS; sample peak {20*np.log10(peak):.2f} dBFS; {near} near-full-scale samples.',
                     'rms_dbfs': rms, 'peak_dbfs': float(20*np.log10(peak)),
                     'near_full_scale_samples': near,
                     'interpretation': 'Amplitude measurements only; no listening or authenticity judgment.',
                     'review_status': 'listening_pending'})
    # Nonoverlapping 20 ms frames. This rule measures quiet, not speech activity.
    frame = int(.02*rate)
    quiet = [level(audio[i:i+frame]) for i in range(0, len(audio), frame)]
    mask = [x is None or x < -40 for x in quiet]
    start = None
    for i, low in enumerate(mask+[False]):
        if low and start is None:
            start = i
        elif not low and start is not None:
            if i-start >= 10:
                rows.append({'start': start*frame/rate, 'end': min(i*frame, len(audio))/rate,
                             'kind': 'low_energy',
                             'observation': 'Consecutive 20 ms frames below -40 dBFS RMS for at least 200 ms.',
                             'rms_dbfs': None, 'peak_dbfs': None, 'near_full_scale_samples': None,
                             'interpretation': 'Possible pause or quiet passage. Not classified as a dropout, breath, or absence of speech.',
                             'review_status': 'listening_pending'})
            start = None
    return sorted(rows, key=lambda row: (row['start'], row['kind']))


def main():
    path = DIRECTORY/'annotation-demo.wav'
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Annotation sample differs from the selected recording')
    audio, rate = sf.read(path, dtype='float64')
    if rate != 16000 or audio.ndim != 1 or len(audio) != 480000 or not np.isfinite(audio).all():
        raise ValueError('Require exactly 30 seconds of finite mono 16 kHz audio')
    rows = annotations(audio, rate)
    result = {
        'schema_version': 'annotation-example-1.0', 'sample': path.name,
        'sample_sha256': EXPECTED, 'duration_seconds': 30, 'sample_rate_hz': rate,
        'scope': 'Reference-informed automated signal annotation demonstration. Not a blind listening submission.',
        'guideline': 'Speech annotation guide 1.1; perceptual fields require listening.',
        'review_status': 'Automated measurements complete; listening review pending',
        'listening_review_completed': False, 'detector_inference_run': False,
        'perceptual_annotation': {
            'intelligibility': 'not_assessed',
            'interference': {key: 'not_assessed' for key in (
                'background_noise', 'compression_artifacts', 'reverb_echo', 'music_hold_audio',
                'audible_clipping', 'dropouts', 'distortion', 'overlap')},
            'synthetic_cues': 'not_assessed', 'human_cues': 'not_assessed',
            'perceived_origin': 'uncertain', 'confidence': None,
            'reason': 'No listening review performed. Reference provenance is separate from perceived origin.',
            'escalation': 'Listen at original speed; confirm intelligibility and channel interference, review quiet intervals, and check whether the 30-second crop truncates a word. Record only cues actually heard.',
        },
        'reference': {'corpus': 'LibriSpeech test-clean', 'source_id': '2094-142345-0010',
                      'reference_origin': 'human', 'reference_origin_basis': 'Corpus provenance, not a listening or detector verdict',
                      'crop_seconds': [0, 30], 'original_duration_seconds': 30.615,
                      'processing': 'First 30 seconds, unmodified PCM samples; no normalization, denoising, or splicing.',
                      'license': 'CC BY 4.0', 'source_url': 'https://www.openslr.org/12/',
                      'attribution': 'LibriSpeech corpus, Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. Excerpt cropped to 30 seconds.',
                      'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                      'privacy': 'Public audiobook excerpt; no personal workstation paths or reviewer identifiers included. Not a guarantee that spoken fiction contains no names.'},
        'source_measurements': measure_source(audio[:, None], rate),
        'annotations': rows,
        'exclusions': 'Demonstration only; excluded from blind review, personal annotation results, reserved evaluation, and agreement metrics.',
    }
    (DIRECTORY/'annotation-demo.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    stream = StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (DIRECTORY/'annotation-demo.csv').write_text(stream.getvalue())
    body = ''.join('<tr><td><button class="seek" data-start="'+str(r['start'])+'" data-end="'+str(r['end'])+'">'+f"{r['start']:.2f}–{r['end']:.2f} s"+'</button></td><td>'+html.escape(r['kind'].replace('_', ' '))+'</td><td>'+html.escape(r['observation'])+'</td><td>'+html.escape(r['interpretation'])+'</td></tr>' for r in rows)
    template = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>30-second annotation example · SignalTrace</title><link rel="stylesheet" href="/styles.css"><link rel="stylesheet" href="/samples/annotation-example.css"></head><body><main class="sample-page"><a href="/">← Back to voice check</a><section class="intro"><div><span class="kicker">ANNOTATION EXAMPLE · 30 SECONDS</span><h1>Observe first. Explain precisely.</h1><p>Automated signal annotations of an English audiobook excerpt. Listening review is pending; these are not personal listening judgments.</p></div></section><section class="workspace"><h2>Original-speed recording</h2><audio id="sample-audio" controls preload="metadata" src="/samples/annotation-demo.wav"></audio><p>30.00 seconds · Mono · 16,000 Hz · Unmodified samples from a longer source.</p><p><a href="/samples/annotation-demo.csv" download>Download interval annotations (CSV)</a> · <a href="/samples/annotation-demo.json" download>Download complete record (JSON)</a> · <a href="/samples/annotation-demo.wav" download>Download recording</a></p></section><section class="workspace"><h2>Recording-level annotation</h2><dl><dt>Measured level</dt><dd>RMS -22.01 dBFS; sample peak -3.70 dBFS. Zero samples at or above 0.999 full scale. This does not establish the absence of audible distortion.</dd><dt>Intelligibility and interference</dt><dd>Not assessed by listening. Noise, compression artifacts, reverb, music, audible clipping, dropouts, distortion, and overlapping voices remain pending.</dd><dt>Perceptual cues and origin</dt><dd>Human and synthetic cues not assessed. Perceived origin: uncertain; confidence: unassigned. No detector inference was used.</dd><dt>Escalation / next review</dt><dd>Listen at original speed, inspect the quiet intervals, and check the cropped endpoint for an interrupted word. Distinguish ordinary pauses from dropouts. Add perceptual labels only after listening.</dd><dt>Reference disclosure</dt><dd>Source corpus identifies genuine audiobook speech. This example is reference-informed and must not count toward blind-review accuracy or independent agreement.</dd></dl></section><section class="workspace"><h2>Timestamped observations</h2><p>Every five-second interval is measured. Additional quiet spans use a fixed -40 dBFS RMS rule over 20 ms frames, for at least 200 ms. Click an interval to play it; ordinary playback remains available above.</p><div class="table-scroll"><table><thead><tr><th>Interval</th><th>Type</th><th>Observation / rule</th><th>Interpretation limit</th></tr></thead><tbody>ROWS</tbody></table></div></section><footer>Source: <a href="https://www.openslr.org/12/">LibriSpeech</a>, Panayotov et al. · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · source 2094-142345-0010 · first 30 seconds of a 30.615-second recording. No processing besides cropping and lossless WAV encoding. Public audiobook text may contain fictional names. This example is excluded from research evaluation and human annotation metrics.</footer></main><script src="/samples/annotation-example.js"></script></body></html>'''
    (DIRECTORY/'annotation-example.html').write_text(template.replace('ROWS', body))
    print(f'Built {len(rows)} timestamped observations; six five-second intervals; listening pending.')


if __name__ == '__main__':
    main()
