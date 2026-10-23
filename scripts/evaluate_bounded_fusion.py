"""Reproduce one rejected fusion candidate on historical development features only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.fusion import fit_logistic, file_score
from scripts.fit_fusion import evaluate, fusion_scorer, training_set


def bounded(row):
    return {**row, 'windows': [{**w, 'pellav2_logit': float(np.clip(w['pellav2_logit'], -5, 5)),
                              'antideepfake_logit': float(np.clip(w['antideepfake_logit'], -5, 5))} for w in row['windows']]}


def candidate_scorer(train):
    beta = fit_logistic(*training_set([bounded(row) for row in train]), l2=1.)
    return lambda row: file_score(beta, np.array([[w['pellav2_logit'], w['antideepfake_logit']] for w in bounded(row)['windows']]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = (ROOT/'data/v4-dev/features.json').read_bytes()
    cache = json.loads(raw)
    spec = json.loads((ROOT/'lab/fusion-v4.json').read_text())
    if sha256(raw).hexdigest() != spec['development']['features_sha256']:
        raise ValueError('Historical development feature hash differs from frozen baseline')
    if cache['models'] != spec['models']:
        raise ValueError('Development features do not match frozen model identities')
    rows = list(cache['rows'].values())
    result = {'schema_version': 'bounded-fusion-experiment-1.0', 'partition': 'historical development only',
        'candidate': 'Fixed clip both component logits to [-5,5]; same logistic L2=1, speaker folds and threshold rule as baseline',
        'selection_status': 'rejected; no production weights, thresholds or files changed',
        'feature_sha256': sha256(raw).hexdigest(),
        'baseline': evaluate(rows, fusion_scorer), 'bounded_candidate': evaluate(rows, candidate_scorer),
        'code_sha256': {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/evaluate_bounded_fusion.py', 'scripts/fit_fusion.py', 'lab/fusion.py')},
        'scope': 'Exploratory reuse of historical development cache. Threshold picked from pooled out-of-fold scores under the historical rule. '
                 'Not separate untouched calibration, nested model selection or independent external validation. No new-model-training claim.'}
    with args.output.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Candidate rejected; see per-condition false-positive counts.')


if __name__ == '__main__':
    main()
