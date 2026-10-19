"""Protocol v3, modern voices: check pellav2 on the calibration split, freeze, then score the test split once.

  calibrate  score the clean calibration split; keep threshold 0.5 if the pre-set bar is met, otherwise
             choose the calibration threshold with the best balanced accuracy (ties nearest 0.5) and re-check.
             Writes data/evaluation/modern-freeze.json only if the bar is met; never rewrites it.
  test       requires the freeze; scores the test split (clean, MP3, phone) with the frozen model and
             threshold, plus older detectors on clean audio for comparison. Writes docs/detector-evaluation.json
             (v3) once; the earlier receipt moves to docs/detector-evaluation-v2.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.evaluation import detection_rate, summarize
from lab.store import digest

MANIFEST = ROOT / 'data/evaluation/modern-manifest.json'
FREEZE = ROOT / 'data/evaluation/modern-freeze.json'
FAILED = ROOT / 'data/evaluation/modern-calibration-failed.json'
RECEIPT = ROOT / 'docs/detector-evaluation.json'
LEGACY = ROOT / 'docs/detector-evaluation-v2.json'
RUNS = ROOT / 'data/evaluation/modern-runs'
MODEL = 'pellav2'
COMPARISON = ('aasist', 'wav2vec2')
BAR = {'min_fake_detection': .95, 'max_real_false_alarms': .05, 'applies_to': 'clean test split',
       'fixed': 'before any modern-voice clip was scored'}
LIMITATIONS = [
    'pellav2 training data is private and may include Chatterbox or F5-TTS output. These results measure fresh clips from popular open generators, not unseen generator families.',
    'Two generators, English only, read speech. ElevenLabs and other commercial voices, voice conversion, conversational speech, and other languages were not tested.',
    'Real speech is LibriSpeech (LibriVox audiobooks, 16 kHz, MP3 history); fakes are generated at 24 kHz. Everything is resampled to 16 kHz before inference, and the MP3 and telephone conditions put both classes through the same channel. LibriSpeech may also be in the detector’s training data, which can make false-alarm rates optimistic.',
    'Chatterbox embeds an imperceptible Perth watermark; the detector was not trained here, so it cannot have learned it from this set.',
    'Wilson intervals assume independent clips. Clips share speakers and texts, so true uncertainty is somewhat larger.',
]


def load_manifest():
    manifest = json.loads(MANIFEST.read_text())
    if digest({k: v for k, v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('Modern-voice manifest changed since it was frozen')
    return manifest


def rates(rows, threshold):
    fakes = [r for r in rows if r['label'] == 'synthetic']
    reals = [r for r in rows if r['label'] == 'human']
    return detection_rate(fakes, threshold), detection_rate(reals, threshold)


def bar_met(rows, threshold):
    fakes, reals = rates(rows, threshold)
    return fakes['rate'] >= BAR['min_fake_detection'] and reals['rate'] <= BAR['max_real_false_alarms']


def choose_threshold(rows):
    """0.5 if it meets the bar; else the best balanced-accuracy cut on these rows, ties nearest 0.5."""
    if bar_met(rows, .5):
        return .5, 'author threshold 0.5 met the bar on calibration'
    scores = sorted({r['score'] for r in rows})
    candidates = [.5] + [(a+b)/2 for a, b in zip(scores, scores[1:])]
    best = max(candidates, key=lambda t: (summarize(rows, t)['balanced_accuracy'], -abs(t-.5)))
    return best, 'best calibration balanced accuracy, ties nearest 0.5, because 0.5 missed the bar'


def score(model, cases, channel='clean'):
    from lab.forensics import analyze, save_report
    rows = []
    for i, case in enumerate(cases, 1):
        source = ROOT / (case['path'] if channel == 'clean' else case['channels'][channel]['path'])
        expected = case['sha256'] if channel == 'clean' else case['channels'][channel]['sha256']
        blob = source.read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f'Audio changed since freezing: {case["id"]} ({channel})')
        report, _ = analyze(blob, model)
        save_report(report, RUNS)
        rows.append({k: case[k] for k in ('id', 'cell', 'label', 'generator', 'source', 'speaker')} |
                    {'channel': channel, 'score': report['assessment']['synthetic_score'],
                     'assessment': report['assessment']['assessment'], 'reasons': report['assessment']['abstention_reasons'],
                     'windows': len(report['windows'])})
        if i % 20 == 0 or i == len(cases):
            print(f'{model.name} {channel} {i}/{len(cases)}', flush=True)
    return rows


def block(rows, threshold):
    fakes, reals = rates(rows, threshold)
    return {'summary': summarize(rows, threshold), 'fakes': fakes, 'reals': reals,
            'by_generator': {g: detection_rate([r for r in rows if r['generator'] == g], threshold)
                             for g in sorted({r['generator'] for r in rows if r['label'] == 'synthetic'})},
            'by_source': {s: detection_rate([r for r in rows if r['label'] == 'human' and r['source'] == s], threshold)
                          for s in sorted({r['source'] for r in rows if r['label'] == 'human'})}}


def calibrate():
    from lab.detectors import load
    if FREEZE.exists():
        print('Calibration is already frozen:', FREEZE.relative_to(ROOT))
        return 0
    manifest = load_manifest()
    model = load(MODEL)
    rows = score(model, [c for c in manifest['cases'] if c['split'] == 'calibration'])
    threshold, rule = choose_threshold(rows)
    result = {'model': MODEL, 'model_manifest': model.manifest, 'manifest_sha256': manifest['sha256'], 'bar': BAR,
              'threshold': threshold, 'rule': rule, 'bar_met_at_0.5': bar_met(rows, .5),
              'calibration': block(rows, threshold) | {'rows': rows}, 'created_at': datetime.now(timezone.utc).isoformat()}
    fakes, reals = result['calibration']['fakes'], result['calibration']['reals']
    print(f'Calibration at threshold {threshold:.4f}: fakes flagged {fakes["flagged_synthetic"]}/{fakes["n"]}, '
          f'real flagged {reals["flagged_synthetic"]}/{reals["n"]}')
    if not bar_met(rows, threshold):
        FAILED.write_text(json.dumps(result, indent=2) + '\n')
        print('STOP: the bar is not met on calibration even after the threshold rule. Nothing frozen; see', FAILED.relative_to(ROOT))
        return 1
    FREEZE.write_text(json.dumps(result, indent=2) + '\n')
    print('Frozen:', FREEZE.relative_to(ROOT), 'threshold', threshold)
    return 0


def test():
    from lab.detectors import installed, load
    if not FREEZE.exists():
        raise ValueError('Run calibrate first; the test split is scored only after the freeze.')
    if RECEIPT.exists() and json.loads(RECEIPT.read_text()).get('version') == '3.0':
        raise ValueError('The v3 test split has already been scored once. A change needs a new protocol version with fresh clips.')
    manifest, freeze = load_manifest(), json.loads(FREEZE.read_text())
    if freeze['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Freeze does not match the manifest')
    threshold = freeze['threshold']
    cases = [c for c in manifest['cases'] if c['split'] == 'test']
    model = load(MODEL)
    if model.manifest != freeze['model_manifest']:
        raise ValueError('Detector files differ from the frozen calibration')
    model.threshold = threshold
    channels = {ch: score(model, cases, ch) for ch in ('clean', 'mp3', 'phone')}
    comparison, models = {}, {MODEL: {**model.manifest, 'parameters': model.parameters, 'name': model.name}}
    for kind in COMPARISON:
        if not installed(kind):
            continue
        other = load(kind)
        models[kind] = {**other.manifest, 'parameters': other.parameters, 'name': other.name}
        comparison[kind] = block(score(other, cases), getattr(other, 'threshold', .5))
    clean = block(channels['clean'], threshold)
    receipt = {'version': '3.0', 'protocol': 'modern voices', 'created_at': datetime.now(timezone.utc).isoformat(),
               'bar': BAR, 'bar_met': bar_met(channels['clean'], threshold), 'selected_model': MODEL,
               'threshold': threshold, 'threshold_rule': freeze['rule'], 'manifest_sha256': manifest['sha256'],
               'freeze_sha256': digest(freeze), 'models': models,
               'calibration': {k: freeze['calibration'][k] for k in ('summary', 'fakes', 'reals', 'by_generator', 'by_source')},
               'test': {**clean, 'by_channel': {ch: {k: v for k, v in block(rows, threshold).items() if k in ('summary', 'fakes', 'reals')}
                                                for ch, rows in channels.items()},
                        'rows': [r for rows in channels.values() for r in rows]},
               'comparison': comparison, 'limitations': LIMITATIONS}
    if RECEIPT.exists() and not LEGACY.exists():
        RECEIPT.replace(LEGACY)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    f, r = clean['fakes'], clean['reals']
    print(f'TEST (clean): fakes flagged {f["flagged_synthetic"]}/{f["n"]}, real flagged {r["flagged_synthetic"]}/{r["n"]}; '
          f'bar {"MET" if receipt["bar_met"] else "NOT MET"}')
    for ch, rows in channels.items():
        cf, cr = rates(rows, threshold)
        print(f'  {ch}: fakes {cf["flagged_synthetic"]}/{cf["n"]}, real false alarms {cr["flagged_synthetic"]}/{cr["n"]}')
    for kind, result in comparison.items():
        print(f'  {kind} (clean): fakes {result["fakes"]["flagged_synthetic"]}/{result["fakes"]["n"]}, '
              f'real false alarms {result["reals"]["flagged_synthetic"]}/{result["reals"]["n"]}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['calibrate', 'test'])
    return {'calibrate': calibrate, 'test': test}[parser.parse_args().stage]()


if __name__ == '__main__':
    raise SystemExit(main())
