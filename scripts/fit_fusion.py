"""Protocol v4: fit the pellav2 + AntiDeepfake fusion on development features and freeze it.

Rule, fixed before the development features existed:
  1. Features per 4 s window: pellav2 logit and AntiDeepfake (fake - real) logit; 3-parameter logistic
     model, L2 = 1, each clip version weighted equally and the two classes weighted equally.
  2. Five-fold cross-validation grouped by speaker; out-of-fold file scores = mean window probability.
  3. Threshold = highest value that still flags >= 98% of out-of-fold development fakes, all channels pooled.
  4. Bar on out-of-fold scores: >= 95% of fakes caught per generator in every channel; false alarms
     <= 2% clean and <= 5% MP3, telephone, and 20 dB noise.
  5. If the bar is missed: stop, write data/evaluation/v4-fusion-failed.json, freeze nothing.
     If met: refit on all development data and write lab/fusion-v4.json once.
The same rule is applied to each model alone for comparison.

Attempt 1 (base, above) missed the bar only on 20 dB noise (11/180 real flagged). Attempt 2, fixed before
running: add a per-window noise index n = clip((40 - D) / 20, 0, 2), where D is the 95th minus 10th
percentile of 20 ms frame levels in dB (lab.fusion.noise_index), plus interactions a·n and b·n
(6 parameters). Everything else in steps 1-5 is unchanged. If attempt 2 also misses the bar, the owner
chose to freeze the base fusion anyway (--model base --accept-miss) and report the miss on the fresh test.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import floor, log
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.evaluation import detection_rate, summarize
from lab.fusion import WEIGHTS, file_score, fit_logistic, sigmoid

FEATURES = ROOT / 'data/v4-dev/features.json'
FAILED = ROOT / 'data/evaluation/v4-fusion-failed.json'
CHANNELS = ('clean', 'mp3', 'phone', 'noise20')
SEED, FOLDS, L2, TARGET_CATCH = 20261019, 5, 1.0, .98
BAR = {'min_fake_detection_each_channel': .95, 'min_heldout_generator_detection': .90,
       'max_false_alarms': {'clean': .02, 'mp3': .05, 'phone': .05, 'noise20': .05},
       'summary': '≥95% of fakes caught per generator in every channel (≥90% for the held-out XTTS-v2 on the test); '
                  'false alarms ≤2% on clean speech and ≤5% on MP3, telephone, and noisy speech'}


NOISE_AWARE = False


def window_features(row):
    if NOISE_AWARE:
        return np.array([[w['pellav2_logit'], w['antideepfake_logit'], w['noise_index']] for w in row['windows']])
    return np.array([[w['pellav2_logit'], w['antideepfake_logit']] for w in row['windows']])


def training_set(rows):
    X, y, weight = [], [], []
    per_class = defaultdict(int)
    for r in rows:
        per_class[r['label']] += 1
    for r in rows:
        f = window_features(r)
        X.append(f)
        y += [r['label'] == 'synthetic'] * len(f)
        # Each clip version counts once; each class carries half the total weight.
        weight += [1 / (len(f) * per_class[r['label']])] * len(f)
    return np.vstack(X), np.array(y, dtype=float), np.array(weight)


def choose_threshold(rows, target=TARGET_CATCH):
    fakes = sorted(r['score'] for r in rows if r['label'] == 'synthetic')
    return fakes[floor((1 - target) * len(fakes))]


def channel_report(rows, threshold):
    report = {}
    for ch in CHANNELS:
        part = [r for r in rows if r['channel'] == ch]
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
        if cell['reals']['rate'] > BAR['max_false_alarms'][ch]:
            failures.append(f'{ch}: {cell["reals"]["flagged_synthetic"]}/{cell["reals"]["n"]} real flagged > {BAR["max_false_alarms"][ch]:.0%}')
    return failures


def folds(rows):
    speakers = sorted({r['speaker'] for r in rows})
    random.Random(SEED).shuffle(speakers)
    return {s: i % FOLDS for i, s in enumerate(speakers)}


def evaluate(rows, scorer):
    """Out-of-fold file scores for one scoring method; `scorer(train_rows)` returns row -> score."""
    fold_of = folds(rows)
    scored = []
    for k in range(FOLDS):
        train = [r for r in rows if fold_of[r['speaker']] != k]
        score = scorer(train)
        scored += [{**{key: r[key] for key in ('id', 'channel', 'label', 'generator', 'source', 'speaker')},
                    'score': score(r), 'assessment': 'n/a'} for r in rows if fold_of[r['speaker']] == k]
    threshold = choose_threshold(scored)
    report = channel_report(scored, threshold)
    return {'threshold': threshold, 'threshold_logit': log(threshold / (1 - threshold)), 'report': report,
            'failures': bar_failures(report), 'summary_pooled': summarize(scored, threshold), 'rows': scored}


def fusion_scorer(train):
    beta = fit_logistic(*training_set(train), l2=L2)
    return lambda r: file_score(beta, window_features(r))


def single_scorer(column):
    return lambda train: (lambda r: float(np.mean(sigmoid(window_features(r)[:, column]))))


def main():
    global NOISE_AWARE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['base', 'noise-aware'], default='base')
    parser.add_argument('--accept-miss', action='store_true', help='freeze even if cross-validation misses the bar (owner decision, recorded)')
    args = parser.parse_args()
    NOISE_AWARE = args.model == 'noise-aware'
    if WEIGHTS.exists():
        print('Fusion already frozen:', WEIGHTS.relative_to(ROOT))
        return 0
    store = json.loads(FEATURES.read_text())
    rows = list(store['rows'].values())
    expected = 360 * len(CHANNELS)
    if len(rows) != expected:
        raise ValueError(f'Development features incomplete: {len(rows)}/{expected}')
    results = {'fusion': evaluate(rows, fusion_scorer),
               'pellav2 alone': evaluate(rows, single_scorer(0)),
               'AntiDeepfake alone': evaluate(rows, single_scorer(1))}
    attempt = 'attempt 2 (noise-aware)' if NOISE_AWARE else 'attempt 1 (base)'
    for name, result in results.items():
        cells = ' | '.join(f'{ch}: fakes {c["fakes"]["flagged_synthetic"]}/{c["fakes"]["n"]} real {c["reals"]["flagged_synthetic"]}/{c["reals"]["n"]}'
                           for ch, c in result['report'].items())
        print(f'{name:18s} t={result["threshold"]:.6f} (logit {result["threshold_logit"]:.2f}) '
              f'{"PASS" if not result["failures"] else "fail"} :: {cells}')
    fusion = results['fusion']
    record = {'protocol': '4.0', 'kind': args.model, 'attempt': attempt, 'rule': __doc__.split('Rule, fixed before the development features existed:\n')[1].strip(),
              'bar': BAR, 'development': {'features_sha256': hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
                                          'v3_manifest_sha256': store['v3_manifest_sha256'], 'clip_versions': len(rows),
                                          'channels': list(CHANNELS)},
              'cross_validation': {name: {k: v for k, v in r.items() if k != 'rows'} for name, r in results.items()},
              'created_at': datetime.now(timezone.utc).isoformat()}
    if fusion['failures'] and not args.accept_miss:
        failed = FAILED.with_name(f'v4-fusion-failed-{args.model}.json')
        failed.write_text(json.dumps(record | {'failures': fusion['failures']}, indent=2) + '\n')
        print('STOP: fusion misses the bar in cross-validation:', '; '.join(fusion['failures']))
        return 1
    record['development_bar_failures'] = fusion['failures']
    beta = fit_logistic(*training_set(rows), l2=L2)
    spec = {'version': '4.0', 'beta': [float(b) for b in beta], 'threshold': fusion['threshold'],
            'threshold_logit': fusion['threshold_logit'], 'models': store['models'], 'l2': L2,
            'features': ['pellav2 logit', 'AntiDeepfake logit (fake - real)'] + (['noise index', 'pellav2 logit × noise', 'AntiDeepfake logit × noise'] if NOISE_AWARE else []),
            'clip': 30.0} | record
    WEIGHTS.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Frozen {WEIGHTS.relative_to(ROOT)}: beta {spec["beta"]}, threshold {fusion["threshold"]!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
