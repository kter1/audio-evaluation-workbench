from lab.store import digest
from scripts.build_modern_set import COUNTS, make_plan, select, sentence


def corpus(prefix, speakers, per_speaker=20):
    rows = []
    for s in range(speakers):
        for u in range(per_speaker):
            utt = f'{prefix}{s:02d}-1-{u:04d}'
            rows.append({'utt': utt, 'speaker': utt.split('-')[0], 'text': f'WORD{s} TEXT{u} I SAID',
                         'path': f'audio/{utt}.flac', 'duration': 3.5 + (u % 10)})
    return rows


def plan():
    return make_plan(corpus('1', 40), corpus('9', 10))


def test_plan_is_speaker_disjoint_matched_and_reproducible():
    p = plan()
    assert len(p['speakers']['calibration']) == 12 and len(p['speakers']['test']) == 28
    assert not set(p['speakers']['calibration']) & set(p['speakers']['test'])
    for case in p['cases']:
        if case['source'] == 'test-clean':
            assert case['speaker'] in p['speakers'][case['split']]
    generators = {(c['split'], c['generator']) for c in p['cases']}
    assert {('calibration', 'chatterbox'), ('calibration', 'f5'), ('test', 'chatterbox'), ('test', 'f5')} <= generators
    refs = set(p['refs'].values())
    reals = {(c['split'], c['source_utt']) for c in p['cases'] if c['label'] == 'human'}
    for case in p['cases']:
        if case['label'] == 'synthetic':
            assert case['source_utt'] != case['ref_utt'] and case['text'] != case['ref_text']
            assert case['ref_utt'] == p['refs'][case['speaker']]
            assert (case['split'], case['source_utt']) in reals  # matched real counterpart
    assert not refs & {u for _, u in reals}  # voice prompts never appear as real test clips
    assert len({c['id'] for c in p['cases']}) == len(p['cases'])
    assert digest(plan()) == digest(p)
    assert sentence('I THINK I SAW IT') == 'I think I saw it.'


def test_selection_skips_failed_gate_in_order_and_reports_shortfalls():
    p = plan()
    gate = {c['id']: {'wer': 0.0, 'hypothesis': c['text']} for c in p['cases']}
    first_fake = next(c for c in p['cases'] if c['cell'] == 'test/f5')
    gate[first_fake['id']] = {'wer': .5, 'hypothesis': 'garbled'}
    missing = next(c for c in p['cases'] if c['cell'] == 'calibration/chatterbox')
    del gate[missing['id']]
    selected, excluded, unused, short = select(p, gate)
    assert not short
    cells = {cell: sum(c['cell'] == cell for c in selected) for cell in COUNTS}
    assert cells == COUNTS
    reasons = {e['id']: e['reason'] for e in excluded}
    assert 'WER 0.500' in reasons[first_fake['id']] and missing['id'] in reasons
    assert first_fake['id'] not in {c['id'] for c in selected}
    assert len(selected) + len(excluded) + len(unused) == len(p['cases'])
    failing = {c['id']: {'wer': 1.0, 'hypothesis': ''} for c in p['cases'] if c['cell'] == 'test/chatterbox'}
    assert select(p, {**gate, **failing})[3] == {'test/chatterbox': 60}


def test_threshold_rule_keeps_author_threshold_when_bar_is_met():
    from scripts.evaluate_modern import bar_met, choose_threshold
    good = [{'label': 'synthetic', 'score': .9, 'assessment': 'x'}]*20 + [{'label': 'human', 'score': .1, 'assessment': 'x'}]*20
    assert bar_met(good, .5) and choose_threshold(good)[0] == .5
    shifted = [{'label': 'synthetic', 'score': .9, 'assessment': 'x'}]*20 + [{'label': 'human', 'score': .6, 'assessment': 'x'}]*20
    assert not bar_met(shifted, .5)
    threshold, rule = choose_threshold(shifted)
    assert threshold == .75 and bar_met(shifted, threshold) and 'missed' in rule
