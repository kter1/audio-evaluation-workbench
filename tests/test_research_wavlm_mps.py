import json

import pytest

from lab.research_wavlm_aasist import ROOT
from lab.research_wavlm_mps import WavLMMPSDetector
from scripts.screen_wavlm_mps import comparison, selected_rows, validate_parity


def score(value=.2, logit=1.4, identity='same'):
    return {'synthetic_score': value, 'native_real_logit': logit,
            'model_input_sha256': identity}


def test_small_numeric_device_rounding_is_allowed():
    assert comparison(score(), score(.20001, 1.4001), score())['passed']


@pytest.mark.parametrize('gpu', [score(.2002), score(.2, 1.402), score(identity='changed')])
def test_probability_logit_or_input_mismatch_blocks_acceleration(gpu):
    assert not comparison(score(), gpu, score())['passed']


def test_binary_flip_blocks_acceleration_even_inside_numeric_tolerance():
    result = comparison(score(.49999), score(.50001), score(.49999))
    assert not result['passed'] and not result['binary_matches']


def test_recomputed_cpu_must_match_frozen_cpu_screen():
    assert not comparison(score(), score(), score(.201))['passed']


def test_fixed_selection_covers_both_labels_all_conditions_and_all_ai_families():
    document = json.loads((ROOT/'docs/core-wavlm-aasist-screen-v1.json').read_text())
    selected = selected_rows(document)
    assert len(selected) == 10
    assert len({row['condition'] for row in selected}) == 5
    assert len({row['group'] for row in selected if row['label'] == 'synthetic'}) == 5
    assert sum(row['label'] == 'human' for row in selected) == 5
    # Reordering the input does not change the preregistered selection.
    document['results'].reverse()
    assert [row['id'] for row in selected_rows(document)] == [row['id'] for row in selected]


def test_absent_or_stale_parity_is_rejected():
    with pytest.raises(ValueError, match='passing parity report'):
        validate_parity({'passed': True}, b'{}')


def test_unavailable_mps_fails_before_checkpoint_access(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='MPS is unavailable'):
        WavLMMPSDetector('/not/a/model/directory')
