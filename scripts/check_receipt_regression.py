"""Re-score a seeded sample of the frozen v4 test with the current environment and compare to the receipt.

Run after any dependency upgrade. The published numbers describe the shipped app only if verdicts are
unchanged and score drift is negligible. This re-scores already-reported clips for a regression check;
it is not a new evaluation and changes no receipt.
"""
import argparse
import hashlib
import json
from math import log
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return log(p / (1 - p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-channel', type=int, default=25)
    args = parser.parse_args()
    from lab.detectors import load
    from lab.forensics import analyze
    receipt = json.loads((ROOT / 'docs/detector-evaluation.json').read_text())
    manifest = json.loads((ROOT / 'data/evaluation/v4-manifest.json').read_text())
    if receipt.get('version') != '4.0' or receipt['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Expected the v4 receipt and its frozen manifest')
    cases = {c['id']: c for c in manifest['cases']}
    rng = random.Random(20261019)
    sample = []
    for channel in manifest['channels']:
        rows = sorted((r for r in receipt['test']['rows'] if r['channel'] == channel), key=lambda r: r['id'])
        sample += rng.sample(rows, min(args.per_channel, len(rows)))
    model, threshold = load('fusion'), receipt['threshold']
    worst_p = worst_z = 0.0
    flips = []
    for i, row in enumerate(sample, 1):
        case = cases[row['id']]
        meta = case if row['channel'] == 'clean' else case['channels'][row['channel']]
        blob = (ROOT / meta['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta['sha256']:
            raise ValueError(f'Audio changed: {row["id"]} ({row["channel"]})')
        score = analyze(blob, model)[0]['assessment']['synthetic_score']
        worst_p = max(worst_p, abs(score - row['score']))
        worst_z = max(worst_z, abs(logit(score) - logit(row['score'])))
        if (score >= threshold) != (row['score'] >= threshold):
            flips.append(f'{row["id"]} {row["channel"]}: {row["score"]:.4f} -> {score:.4f}')
        if i % 25 == 0:
            print(f'regression {i}/{len(sample)}', flush=True)
    print(f'{len(sample)} clip versions re-scored: largest |Δscore| {worst_p:.2e}, largest |Δlogit| {worst_z:.3f}, '
          f'verdict flips {len(flips)}')
    for flip in flips:
        print('  FLIP', flip)
    return 1 if flips else 0


if __name__ == '__main__':
    raise SystemExit(main())
