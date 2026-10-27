"""Build the fixed local annotation demonstration without model inference."""
from pathlib import Path
from hashlib import sha256
from io import StringIO
import csv
import html
import json
import re
from difflib import SequenceMatcher
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.source_measurements import measure_source

EXPECTED = '299c3e4a38abd3e74c61b39ec52824f406f75c62673b0220db5167b3aaabb640'
DIRECTORY = ROOT/'studio/dist/samples'
REFERENCE_TEXT = "HETTY SORREL OFTEN TOOK THE OPPORTUNITY WHEN HER AUNT'S BACK WAS TURNED OF LOOKING AT THE PLEASING REFLECTION OF HERSELF IN THOSE POLISHED SURFACES FOR THE OAK TABLE WAS USUALLY TURNED UP LIKE A SCREEN AND WAS MORE FOR ORNAMENT THAN FOR USE AND SHE COULD SEE HERSELF SOMETIMES IN THE GREAT ROUND PEWTER DISHES THAT WERE RANGED ON THE SHELVES ABOVE THE LONG DEAL DINNER TABLE OR IN THE HOBS OF THE GRATE WHICH ALWAYS SHONE LIKE JASPER"


def transcript_review(record, sample_sha256):
    if record['sample_sha256'] != sample_sha256:
        raise ValueError('Transcript belongs to a different recording')
    words = record['words']
    hypothesis, indexed = [], []
    previous_start = 0
    for word in words:
        start, end = word['start'], word['end']
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not np.isfinite([start, end]).all() or not 0 <= start <= end <= 30
                or start < previous_start):
            raise ValueError('Invalid model-estimated word timestamps')
        previous_start = start
        for token in re.findall(r"[a-z0-9']+", word['text'].lower()):
            hypothesis.append(token)
            indexed.append(word)
    if not hypothesis:
        raise ValueError('Transcript has no words')
    reference = re.findall(r"[a-z0-9']+", REFERENCE_TEXT.lower())
    differences = []
    for kind, a, b, c, d in SequenceMatcher(None, reference, hypothesis, autojunk=False).get_opcodes():
        if kind == 'equal':
            continue
        first, last = indexed[min(c, len(indexed)-1)], indexed[min(max(c, d-1), len(indexed)-1)]
        differences.append({'start': first['start'], 'end': last['end'],
                            'kind': 'transcript_difference', 'status': 'review_required',
                            'model_text': ' '.join(hypothesis[c:d]) or '[omitted]',
                            'reference_text': ' '.join(reference[a:b]) or '[not in source text]',
                            'basis': 'Word-sequence comparison against full-source corpus text. Timing is model-estimated; confirm the cropped audio by listening.'})
    phrases = []
    for i in range(0, len(words), 12):
        group = words[i:i+12]
        phrases.append({'start': group[0]['start'], 'end': group[-1]['end'],
                        'text': ' '.join(w['text'] for w in group)})
    return differences, phrases


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
                     'review_status': 'measured'})
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
                             'review_status': 'context_review'})
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
    transcription = json.loads((DIRECTORY/'annotation-transcript.json').read_text())
    differences, phrases = transcript_review(transcription, EXPECTED)
    result = {
        'schema_version': 'annotation-example-1.1', 'sample': path.name,
        'sample_sha256': EXPECTED, 'duration_seconds': 30, 'sample_rate_hz': rate,
        'scope': 'Reference-informed automated signal annotation demonstration. Not a blind listening submission.',
        'guideline': 'Speech annotation guide 1.1; perceptual fields require listening.',
        'review_status': 'Measurements and local transcription complete; targeted review items identified',
        'transcription': transcription, 'transcript_review': differences,
        'review_summary': {'transcript_differences': len(differences),
                           'quiet_spans': sum(r['kind'] == 'low_energy' for r in rows),
                           'perceptual_review': 'not_run'},
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
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    (DIRECTORY/'annotation-demo.csv').write_text(stream.getvalue())
    with (DIRECTORY/'annotation-review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]) if differences else ['start', 'end', 'kind', 'status', 'model_text', 'reference_text', 'basis'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(differences)
    def replay(start, end, label, context=False):
        if context:
            start, end = max(0, start-.3), min(30, end+.3)
        return f'<button class="seek" data-start="{start}" data-end="{end}">{html.escape(label)}</button>'
    template = (ROOT/'scripts/templates/annotation-example.html').read_text()
    intervals = [r for r in rows if r['kind'] == 'interval_review']
    quiet = [r for r in rows if r['kind'] == 'low_energy']
    values = {
        'DIFFCOUNT': str(len(differences)), 'QUIETCOUNT': str(len(quiet)),
        'WORDCOUNT': str(len(transcription['words'])),
        'MODELNAME': html.escape(transcription['model']), 'MODELREV': html.escape(transcription['revision']),
        'PHRASES': ''.join(replay(p['start'], p['end'], p['text']) for p in phrases),
        'DIFFERENCES': ''.join('<tr><td>'+replay(d['start'], d['end'], f"{d['start']:.2f}–{d['end']:.2f} s", True)+'</td><td>'+html.escape(d['model_text'])+'</td><td>'+html.escape(d['reference_text'])+'</td></tr>' for d in differences),
        'QUIETBUTTONS': ''.join(replay(r['start'], r['end'], f"{r['start']:.2f}–{r['end']:.2f} s", True) for r in quiet),
        'MEASUREMENTS': ''.join('<tr><td>'+replay(r['start'], r['end'], f"{r['start']:.0f}–{r['end']:.0f} s")+f"</td><td>{r['rms_dbfs']:.2f}</td><td>{r['peak_dbfs']:.2f}</td><td>{r['near_full_scale_samples']}</td></tr>" for r in intervals),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    (DIRECTORY/'annotation-example.html').write_text(template)
    print(f'Built {len(differences)} targeted transcript checks; {len(quiet)} grouped quiet spans; {len(phrases)} playback passages.')



if __name__ == '__main__':
    main()
