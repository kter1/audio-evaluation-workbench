from copy import deepcopy

import pytest

from scripts.calibrate_complementary_detector import calibrate


def rows():
    return [{'id':f'{group}-{label}-{condition}', 'group':group, 'label':label,
             'condition':condition, 'synthetic_score':.1 if label == 'human' else .9}
            for group in ('a','b','c','d') for label in ('human','synthetic')
            for condition in ('clean','noise20')]


def test_evaluation_scores_cannot_select_threshold_and_groups_stay_together():
    source = rows()
    first = calibrate(source)
    assert not set(first['calibration_groups']) & set(first['evaluation_groups'])
    changed = deepcopy(source)
    for row in changed:
        if row['group'] in first['evaluation_groups']:
            row['synthetic_score'] = 1.
    second = calibrate(changed)
    assert first['threshold'] == second['threshold']
    assert second['evaluation']['noise20']['human_false_alerts'] == 2


def test_duplicate_and_nonfinite_scores_are_rejected():
    source = rows()
    with pytest.raises(ValueError, match='unique'):
        calibrate(source+[source[0]])
    source[0]['synthetic_score'] = float('nan')
    with pytest.raises(ValueError, match='Invalid'):
        calibrate(source)
