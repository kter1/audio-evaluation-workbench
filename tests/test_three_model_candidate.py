from copy import deepcopy

import pytest

from scripts.analyze_three_model_candidate import compare


@pytest.fixture
def paired():
    baseline_rows, candidate_rows = [], []
    for channel in ('clean', 'mp3', 'phone', 'noise20'):
        for index in range(320):
            label = 'human' if index < 200 else 'synthetic'
            score = .01 if label == 'human' else .9
            record_id = f'original-{index:03d}'
            baseline_rows.append({'id': record_id, 'channel': channel, 'label': label,
                                  'score': score})
            candidate_rows.append({'source_id': record_id, 'condition': channel, 'label': label,
                'synthetic_score': score, 'group': f'speaker-{index % 73}',
                'generator': 'bonafide' if label == 'human' else ('chatterbox', 'f5', 'xtts')[index % 3]})
    candidate = {'case_count': 1280, 'original_clip_count': 320, 'threshold': .5,
        'bounds': {'complete_manifest_run': True},
        'model': {'revision': 'cfd11365b02167214dc3b392378809fef6ca3cfa'},
        'manifest_self_digest': 'a'*64, 'results': candidate_rows}
    baseline = {'version': '4.0', 'manifest_sha256': 'a'*64, 'threshold': .20960105988114733,
                'test': {'rows': baseline_rows}}
    return candidate, baseline


def test_same_decisions_pass_only_spent_false_alert_gate(paired):
    result = compare(*paired)
    assert result['passes_no_additional_false_alerts_gate'] is True
    assert 'no production promotion or independent validation' in result['decision']
    assert result['source_groups'] == 73
    for summary in result['summaries'].values():
        assert summary['human'] == 200 and summary['synthetic'] == 120
        assert summary['baseline_ai_detected'] == summary['candidate_ai_detected'] == 120
        assert summary['additional_human_false_alerts'] == 0


@pytest.mark.parametrize('condition', ['clean', 'mp3', 'phone', 'noise20'])
def test_one_new_genuine_false_alert_rejects_even_with_extra_synthetic_catch(paired, condition):
    candidate, baseline = paired
    human = next(r for r in candidate['results'] if r['condition'] == condition and r['label'] == 'human')
    human['synthetic_score'] = .9
    # The OR rescues a missed synthetic example too; that gain cannot waive the gate.
    fake = next(r for r in baseline['test']['rows'] if r['channel'] == condition and r['label'] == 'synthetic')
    fake['score'] = .01
    result = compare(candidate, baseline)
    assert result['passes_no_additional_false_alerts_gate'] is False
    assert result['decision'].startswith('Reject OR candidate')
    assert result['summaries'][condition]['additional_human_false_alerts'] == 1
    assert result['summaries'][condition]['incremental_ai_detected'] == 1
    assert all(s['additional_human_false_alerts'] == 0 for ch, s in result['summaries'].items() if ch != condition)


def test_existing_false_alert_is_not_mistaken_for_an_additional_error(paired):
    candidate, baseline = paired
    candidate['results'][0]['synthetic_score'] = .9
    baseline['test']['rows'][0]['score'] = .9
    result = compare(candidate, baseline)
    summary = result['summaries']['clean']
    assert summary['baseline_human_false_alerts'] == summary['candidate_human_false_alerts'] == 1
    assert summary['additional_human_false_alerts'] == 0
    assert result['passes_no_additional_false_alerts_gate'] is True


@pytest.mark.parametrize('which', ['candidate', 'baseline'])
def test_duplicate_input_rows_cannot_be_silently_collapsed(paired, which):
    candidate, baseline = paired
    rows = candidate['results'] if which == 'candidate' else baseline['test']['rows']
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        compare(candidate, baseline)


def test_mismatched_identity_coverage_is_rejected(paired):
    candidate, baseline = paired
    candidate['results'][0]['source_id'] = 'different-source'
    with pytest.raises(ValueError, match='identities'):
        compare(candidate, baseline)


def test_paired_reference_label_change_is_rejected(paired):
    candidate, baseline = paired
    candidate['results'][0]['label'] = 'synthetic'
    with pytest.raises(ValueError, match='labels'):
        compare(candidate, baseline)


def test_matching_invalid_labels_cannot_escape_class_denominators(paired):
    candidate, baseline = paired
    candidate['results'][0]['label'] = 'unknown'
    baseline['test']['rows'][0]['label'] = 'unknown'
    with pytest.raises(ValueError):
        compare(candidate, baseline)


@pytest.mark.parametrize('which', ['candidate', 'baseline'])
@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.01, 1.01])
def test_invalid_raw_scores_cannot_produce_a_pass(paired, which, score):
    candidate, baseline = paired
    if which == 'candidate':
        candidate['results'][0]['synthetic_score'] = score
    else:
        baseline['test']['rows'][0]['score'] = score
    with pytest.raises(ValueError):
        compare(candidate, baseline)


@pytest.mark.parametrize('change', ['debug', 'threshold', 'revision', 'manifest'])
def test_only_frozen_complete_paired_candidate_is_accepted(paired, change):
    candidate, baseline = paired
    if change == 'debug':
        candidate['bounds']['complete_manifest_run'] = False
    elif change == 'threshold':
        candidate['threshold'] = .4
    elif change == 'revision':
        candidate['model']['revision'] = 'other-model'
    else:
        candidate['manifest_self_digest'] = 'b'*64
    with pytest.raises(ValueError, match='complete frozen paired'):
        compare(candidate, baseline)
