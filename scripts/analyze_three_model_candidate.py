"""Evaluate the frozen raw OR candidate on complete paired regression scores."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare(candidate, baseline):
    if (candidate.get('case_count') != 1280 or candidate.get('original_clip_count') != 320
            or candidate.get('threshold') != .5
            or not candidate.get('bounds', {}).get('complete_manifest_run')
            or candidate.get('model', {}).get('revision') != 'cfd11365b02167214dc3b392378809fef6ca3cfa'
            or baseline.get('version') != '4.0'
            or candidate['manifest_self_digest'] != baseline['manifest_sha256']):
        raise ValueError('Require complete frozen paired v4 regression and fixed candidate')
    old_rows, current_rows = baseline['test']['rows'], candidate['results']
    if len(old_rows) != 1280 or len(current_rows) != 1280:
        raise ValueError('Require exactly 1280 rows on each side, without duplicates')
    threshold = baseline.get('threshold')
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('Invalid baseline threshold')
    for rows, score_name in ((old_rows, 'score'), (current_rows, 'synthetic_score')):
        for row in rows:
            score = row.get(score_name)
            if (row.get('label') not in ('human', 'synthetic')
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid regression label or score')
    old = {(r['id'], r['channel']): r for r in old_rows}
    current = {(r['source_id'], r['condition']): r for r in current_rows}
    if len(old) != 1280 or len(current) != 1280 or set(old) != set(current):
        raise ValueError('Regression identities must match exactly once')
    if ({key[1] for key in old} != {'clean', 'mp3', 'phone', 'noise20'}
            or len({key[0] for key in old}) != 320):
        raise ValueError('Require all four declared conditions and 320 original identities')
    for condition in ('clean', 'mp3', 'phone', 'noise20'):
        for label, required in (('human', 200), ('synthetic', 120)):
            if sum(r['channel'] == condition and r['label'] == label for r in old_rows) != required:
                raise ValueError('Require exact frozen label counts in every condition')
    rows = []
    for key, row in current.items():
        previous = old[key]
        if row['label'] != previous['label']:
            raise ValueError('Paired labels differ')
        before = previous['score'] >= baseline['threshold']
        new = row['synthetic_score'] >= .5
        rows.append({'id': key[0], 'condition': key[1], 'label': row['label'],
                     'group': row['group'], 'generator': row['generator'],
                     'baseline_flag': before, 'wavlm_flag': new,
                     'candidate_flag': before or new})
    summaries = {}
    for condition in ('clean', 'mp3', 'phone', 'noise20'):
        part = [r for r in rows if r['condition'] == condition]
        human = [r for r in part if r['label'] == 'human']
        fake = [r for r in part if r['label'] == 'synthetic']
        summaries[condition] = {
            'human': len(human), 'synthetic': len(fake),
            'baseline_ai_detected': sum(r['baseline_flag'] for r in fake),
            'wavlm_ai_detected': sum(r['wavlm_flag'] for r in fake),
            'candidate_ai_detected': sum(r['candidate_flag'] for r in fake),
            'incremental_ai_detected': sum(r['candidate_flag'] and not r['baseline_flag'] for r in fake),
            'baseline_human_false_alerts': sum(r['baseline_flag'] for r in human),
            'wavlm_human_false_alerts': sum(r['wavlm_flag'] for r in human),
            'candidate_human_false_alerts': sum(r['candidate_flag'] for r in human),
            'additional_human_false_alerts': sum(r['candidate_flag'] and not r['baseline_flag'] for r in human),
        }
    rejected = any(s['additional_human_false_alerts'] for s in summaries.values())
    return {'summaries': summaries, 'paired_raw_decisions': rows,
            'source_groups': len({r['group'] for r in rows}),
            'passes_no_additional_false_alerts_gate': not rejected,
            'decision': 'Reject OR candidate: additional genuine false alerts' if rejected else
                'Passes this spent-regression raw false-alert check only; no production promotion or independent validation',
            'scope': 'Raw file-level binary OR, fixed native thresholds. No product quality, abstention or stability assessment is inferred. '
                     'Post-hoc development candidate tested on spent regression; source-file comparison uses different native contexts. '
                     'No probability, independent superiority, low-population-risk or world-class claim. '
                     'The frozen OR cannot lose baseline raw detections; it can add false alerts.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    baseline_path = ROOT/'docs/detector-evaluation.json'
    spec_path = ROOT/'docs/three-model-candidate-v1.json'
    raw = args.candidate.read_bytes()
    baseline_raw = baseline_path.read_bytes()
    result = compare(json.loads(raw), json.loads(baseline_raw))
    result.update({'candidate_sha256': sha256(raw).hexdigest(),
                   'baseline_sha256': sha256(baseline_raw).hexdigest(),
                   'frozen_rule_sha256': sha256(spec_path.read_bytes()).hexdigest(),
                   'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest()})
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['decision'])
    for name, summary in result['summaries'].items():
        print(name, summary)


if __name__ == '__main__':
    main()
