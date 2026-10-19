"""Protocol v4: move pellav2's threshold using already-seen v3 data, then test once on fresh speakers.

  calibrate  pool the v3 calibration rows (clean) and the spent v3 test rows (clean, MP3, phone); choose the
             highest threshold that still flags >= 98% of those fakes; check the bar per channel; write
             data/evaluation/v4-freeze.json once. No inference runs; no v4 clip is touched.
             (This threshold-only attempt missed its bar; the record stays reproducible.)
  test       score the fresh v4 manifest (clean, MP3, phone, 20 dB noise) once with the frozen fusion from
             scripts/fit_fusion.py and write docs/detector-evaluation.json (v4); the v3 receipt moves to
             docs/detector-evaluation-v3.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import floor, log
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.evaluation import detection_rate, summarize
from lab.store import digest

V3_RECEIPT = ROOT / 'docs/detector-evaluation.json'
V3_ARCHIVE = ROOT / 'docs/detector-evaluation-v3.json'
V3_FREEZE = ROOT / 'data/evaluation/modern-freeze.json'
FREEZE = ROOT / 'data/evaluation/v4-freeze.json'
FAILED = ROOT / 'data/evaluation/v4-calibration-failed.json'
MANIFEST = ROOT / 'data/evaluation/v4-manifest.json'
RUNS = ROOT / 'data/evaluation/v4-runs'
CHANNELS = ('clean', 'mp3', 'phone')
TARGET_CATCH = .98
BAR = {'min_fake_detection_each_channel': .95, 'min_heldout_generator_detection': .90,
       'max_false_alarms': {'clean': .05, 'mp3': .10, 'phone': .10},
       'fixed': 'before any v4 clip was generated or scored'}
LIMITATIONS = [
    'Fusion weights and threshold were fitted on v3 clips of Chatterbox-Turbo and F5-TTS, which pellav2 may also have seen in training. XTTS-v2 is held out from all development to test generalization; one held-out generator is still a narrow check.',
    'The combined detector inherits AntiDeepfake’s non-commercial license (CC BY-NC-SA 4.0).',
    'Channel conditions are simulated (MP3, G.711 µ-law at 8 kHz, white noise at 20 dB SNR), not recorded live calls.',
    'English read speech (LibriSpeech dev sets, LibriVox audiobooks with MP3 history). Conversational, accented, and non-English speech, and real-world noisy recordings (as opposed to added white noise), were not tested.',
    'Fakes scoring below the 0.2096 threshold are reported as human; the abstention band (±2 on the fused-logit scale) is an engineering rule, not a calibrated confidence interval.',
    'Wilson intervals assume independent clips; clips share speakers and texts, so true uncertainty is somewhat larger.',
]


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return log(p / (1 - p))


def calibration_rows():
    v3 = json.loads((V3_ARCHIVE if V3_ARCHIVE.exists() else V3_RECEIPT).read_text())
    if v3.get('version') != '3.0':
        raise ValueError('Expected the v3 receipt as calibration data')
    freeze = json.loads(V3_FREEZE.read_text())
    return v3['test']['rows'] + freeze['calibration']['rows'], v3, freeze


def choose_threshold(rows, target=TARGET_CATCH):
    """Highest threshold that still flags at least `target` of the fakes (ties keep the fake flagged)."""
    fakes = sorted(r['score'] for r in rows if r['label'] == 'synthetic')
    return fakes[floor((1 - target) * len(fakes))]


def channel_report(rows, threshold):
    report = {}
    for ch in CHANNELS:
        part = [r for r in rows if r['channel'] == ch]
        if not part:
            continue
        fakes = [r for r in part if r['label'] == 'synthetic']
        reals = [r for r in part if r['label'] == 'human']
        report[ch] = {'fakes': detection_rate(fakes, threshold), 'reals': detection_rate(reals, threshold),
                      'by_generator': {g: detection_rate([r for r in fakes if r['generator'] == g], threshold)
                                       for g in sorted({r['generator'] for r in fakes})},
                      'by_source': {s: detection_rate([r for r in reals if r['source'] == s], threshold)
                                    for s in sorted({r['source'] for r in reals})}}
    return report


def bar_failures(report, heldout=None):
    failures = []
    for ch, cell in report.items():
        for g, rate in cell['by_generator'].items():
            limit = BAR['min_heldout_generator_detection'] if g == heldout else BAR['min_fake_detection_each_channel']
            if rate['rate'] < limit:
                failures.append(f'{ch}/{g}: {rate["flagged_synthetic"]}/{rate["n"]} fakes caught < {limit:.0%}')
        if cell['reals']['n'] and cell['reals']['rate'] > BAR['max_false_alarms'][ch]:
            failures.append(f'{ch}: {cell["reals"]["flagged_synthetic"]}/{cell["reals"]["n"]} real flagged > {BAR["max_false_alarms"][ch]:.0%}')
    return failures


def calibrate():
    if FREEZE.exists():
        print('v4 threshold already frozen:', FREEZE.relative_to(ROOT))
        return 0
    rows, v3, v3_freeze = calibration_rows()
    threshold = choose_threshold(rows)
    report = channel_report(rows, threshold)
    failures = bar_failures(report)
    result = {'protocol': '4.0', 'model': 'pellav2', 'model_manifest': v3_freeze['model_manifest'],
              'threshold': threshold, 'threshold_logit': logit(threshold),
              'rule': f'highest threshold flagging >= {TARGET_CATCH:.0%} of pooled calibration fakes (all channels)',
              'calibration_data': {'v3_test_rows': len(v3['test']['rows']), 'v3_calibration_rows': len(v3_freeze['calibration']['rows']),
                                   'v3_receipt_sha256': digest(v3), 'v3_manifest_sha256': v3['manifest_sha256'],
                                   'note': 'v3 test results were already reported; v4 reuses them only as calibration data'},
              'bar': BAR, 'calibration': report, 'summary_pooled': summarize(rows, threshold),
              'created_at': datetime.now(timezone.utc).isoformat()}
    for ch, cell in report.items():
        print(f'{ch}: fakes {cell["fakes"]["flagged_synthetic"]}/{cell["fakes"]["n"]}, real flagged {cell["reals"]["flagged_synthetic"]}/{cell["reals"]["n"]}')
    print(f'threshold {threshold!r} (logit {logit(threshold):.3f})')
    if failures:
        FAILED.write_text(json.dumps(result | {'failures': failures}, indent=2) + '\n')
        print('STOP: calibration misses the bar:', '; '.join(failures))
        return 1
    FREEZE.write_text(json.dumps(result, indent=2) + '\n')
    print('Frozen:', FREEZE.relative_to(ROOT))
    return 0


def test():
    """Score the frozen v4 manifest once with the frozen fusion; component scores give single-model baselines."""
    from lab.forensics import analyze, save_report
    from lab.fusion import WEIGHTS, load_spec
    from lab.detectors import load
    from scripts import fit_fusion
    if not WEIGHTS.exists():
        raise ValueError('Freeze the fusion first: scripts/fit_fusion.py')
    if V3_RECEIPT.exists() and json.loads(V3_RECEIPT.read_text()).get('version') == '4.0':
        raise ValueError('The v4 test has already been scored once.')
    spec, fusion_sha = load_spec(WEIGHTS)
    manifest = json.loads(MANIFEST.read_text())
    if digest({k: v for k, v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('v4 manifest changed since it was frozen')
    model = load('fusion')
    rows = []
    for channel in manifest['channels']:
        for i, case in enumerate(manifest['cases'], 1):
            source = ROOT / (case['path'] if channel == 'clean' else case['channels'][channel]['path'])
            expected = case['sha256'] if channel == 'clean' else case['channels'][channel]['sha256']
            blob = source.read_bytes()
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError(f'Audio changed since freezing: {case["id"]} ({channel})')
            report, _ = analyze(blob, model)
            save_report(report, RUNS)
            parts = {name: float(np.mean([w['components'][name] for w in report['windows']])) for name in ('pellav2', 'antideepfake')}
            rows.append({k: case[k] for k in ('id', 'cell', 'label', 'generator', 'source', 'speaker')} |
                        {'channel': channel, 'score': report['assessment']['synthetic_score'], 'components': parts,
                         'assessment': report['assessment']['assessment'], 'reasons': report['assessment']['abstention_reasons'],
                         'windows': len(report['windows'])})
            if i % 40 == 0 or i == len(manifest['cases']):
                print(f'fusion {channel} {i}/{len(manifest["cases"])}', flush=True)
    threshold, heldout = spec['threshold'], manifest['heldout_generator']
    fit_fusion.CHANNELS = tuple(manifest['channels'])
    report = fit_fusion.channel_report(rows, threshold)
    failures = fit_fusion.bar_failures(report, heldout=heldout)

    def alone(name, cut):
        single = [{**r, 'score': r['components'][name]} for r in rows]
        return {'threshold': cut, 'by_channel': {ch: {k: c[k] for k in ('fakes', 'reals', 'by_generator')}
                                                 for ch, c in fit_fusion.channel_report(single, cut).items()}}
    cv = spec['cross_validation']
    clean = report['clean']
    receipt = {'version': '4.0', 'protocol': 'modern voices, two-detector fusion', 'created_at': datetime.now(timezone.utc).isoformat(),
               'selected_model': 'fusion', 'threshold': threshold, 'threshold_logit': spec['threshold_logit'],
               'threshold_rule': spec['rule'], 'bar': spec['bar'], 'bar_met': not failures, 'bar_failures': failures,
               'heldout_generator': heldout, 'manifest_sha256': manifest['sha256'], 'fusion_sha256': fusion_sha,
               'fusion': {'beta': spec['beta'], 'features': spec['features']}, 'models': spec['models'],
               'development': {'clip_versions': spec['development']['clip_versions'], 'cross_validation': cv['fusion']['report'],
                               'threshold_only_attempt': 'data/evaluation/v4-calibration-failed.json'},
               'test': {'fakes': clean['fakes'], 'reals': clean['reals'], 'by_generator': clean['by_generator'],
                        'by_source': clean['by_source'], 'summary': summarize([r for r in rows if r['channel'] == 'clean'], threshold),
                        'by_channel': report, 'rows': rows},
               'comparison': {'pellav2 alone (same rule)': alone('pellav2', cv['pellav2 alone']['threshold']),
                              'AntiDeepfake alone (same rule)': alone('antideepfake', cv['AntiDeepfake alone']['threshold']),
                              'pellav2 at authors’ 0.5': alone('pellav2', .5)},
               'limitations': LIMITATIONS}
    if not V3_ARCHIVE.exists():
        V3_RECEIPT.replace(V3_ARCHIVE)
    V3_RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    for ch, cell in report.items():
        gens = ', '.join(f'{g} {c["flagged_synthetic"]}/{c["n"]}' for g, c in cell['by_generator'].items())
        print(f'{ch}: {gens}; real flagged {cell["reals"]["flagged_synthetic"]}/{cell["reals"]["n"]}')
    for name, result in receipt['comparison'].items():
        print(f'  {name}: ' + ' | '.join(f'{ch} fakes {c["fakes"]["flagged_synthetic"]}/{c["fakes"]["n"]} real {c["reals"]["flagged_synthetic"]}/{c["reals"]["n"]}'
                                         for ch, c in result['by_channel'].items()))
    print('BAR', 'MET' if not failures else 'NOT MET: ' + '; '.join(failures))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['calibrate', 'test'])
    return {'calibrate': calibrate, 'test': test}[parser.parse_args().stage]()


if __name__ == '__main__':
    raise SystemExit(main())
