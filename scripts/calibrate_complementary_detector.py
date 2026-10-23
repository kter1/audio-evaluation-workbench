"""Retrospective group-disjoint calibration; never changes production thresholds."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


def calibrate(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Require nonempty unique scored rows')
    for r in rows:
        if (not r.get('group') or r['label'] not in ('human', 'synthetic')
                or not math.isfinite(r['synthetic_score']) or not 0 <= r['synthetic_score'] <= 1):
            raise ValueError('Invalid scored row')
    groups = sorted({r['group'] for r in rows},
                    key=lambda g: sha256(('complement-calibration-20261023:'+g).encode()).hexdigest())
    if len(groups) < 4:
        raise ValueError('Need at least four connected groups')
    calibration_groups = set(groups[:len(groups)//2])
    fitting = [r for r in rows if r['group'] in calibration_groups]
    evaluation = [r for r in rows if r['group'] not in calibration_groups]
    humans = [r['synthetic_score'] for r in fitting if r['label'] == 'human']
    if not humans or max(humans) >= 1:
        raise ValueError('Cannot establish a finite zero-calibration-alert threshold')
    threshold = math.nextafter(max(humans), 1.)
    def summary(part):
        result = {}
        for condition in sorted({r['condition'] for r in part}):
            cell = [r for r in part if r['condition'] == condition]
            human = [r for r in cell if r['label'] == 'human']
            fake = [r for r in cell if r['label'] == 'synthetic']
            if not human or not fake:
                raise ValueError('Both classes required in every partition/condition')
            result[condition] = {'human': len(human), 'synthetic': len(fake),
                'human_false_alerts': sum(r['synthetic_score'] >= threshold for r in human),
                'ai_detected': sum(r['synthetic_score'] >= threshold for r in fake)}
        return result
    human_groups = {r['group'] for r in fitting if r['label'] == 'human'}
    return {'threshold': threshold, 'threshold_rule': 'Next representable score above maximum genuine calibration score, pooled conditions; no synthetic labels used to select threshold',
        'calibration_groups': sorted(calibration_groups), 'evaluation_groups': sorted(set(groups)-calibration_groups),
        'calibration': summary(fitting), 'evaluation': summary(evaluation),
        'zero_calibration_error_one_sided95_group_upper_bound': 1-.05**(1/len(human_groups)),
        'scope': 'Retrospective partition of previously examined regression data. Groups and derivatives stay together. '
                 'Evaluation scores do not select the threshold, but prior model selection used these data. '
                 'This is not untouched independent validation, calibrated authenticity probability, or production promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    source = json.loads(raw)
    if source.get('case_count') != 1280 or source.get('original_clip_count') != 320:
        parser.error('Require the complete 320-original regression')
    result = calibrate(source['results'])
    result.update({'input_sha256': sha256(raw).hexdigest(), 'model': source['model'],
                   'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest()})
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('model','calibration_groups','evaluation_groups')}, indent=2))


if __name__ == '__main__':
    main()
