import json

import numpy as np
import pytest
import soundfile as sf

from lab.challenge import prepare_challenge
from lab.reporting import challenge_summary, manifest_export, review_rows
from lab.store import Store, digest
from lab.workflow import progress
from conftest import APP, payload


def make_fixture(root):
    """Generated tones standing in for speech; real at 16 kHz, fakes at 24 kHz like the generators."""
    cases, rows = [], []
    specs = [('human', 'bonafide', 16000, 12), ('synthetic', 'chatterbox', 24000, 6), ('synthetic', 'f5', 24000, 6)]
    for label, generator, rate, n in specs:
        for k in range(n):
            cid = f'tes-{generator}-{k:03d}'
            path = root / f'{cid}.wav'
            t = np.arange(3*rate)/rate
            sf.write(path, (0.05*(k+1)*np.sin(2*np.pi*(180+20*k)*t)).astype(np.float32), rate, subtype='PCM_16')
            cases.append({'id': cid, 'split': 'test', 'label': label, 'generator': generator,
                          'source': 'test-clean', 'speaker': f'{k % 6}', 'path': path.name, 'sha256': cid})
            score = {'bonafide': .7 if k == 0 else .1, 'chatterbox': .9, 'f5': .2 if k == 0 else .95}[generator]
            rows += [{'id': cid, 'channel': 'clean', 'score': score}, {'id': cid, 'channel': 'phone', 'score': .5}]
    manifest = {'version': '3.0', 'cases': cases}
    manifest['sha256'] = digest(manifest)
    receipt = {'version': '3.0', 'manifest_sha256': manifest['sha256'], 'threshold': .5, 'selected_model': 'pellav2',
               'test': {'rows': rows}}
    return manifest, receipt


@pytest.fixture
def challenge(tmp_path):
    manifest, receipt = make_fixture(tmp_path)
    store = Store(tmp_path/'challenge.sqlite')
    study = prepare_challenge(store, manifest, receipt, 'tester', root=tmp_path)
    yield store, study, manifest, receipt
    store.close()


def test_selection_is_balanced_seeded_and_blind(challenge, tmp_path):
    store, study, manifest, receipt = challenge
    assert store.setting('dataset_kind') == 'challenge'
    assert len(study['selected']) == 20
    assert sorted(s['generator'] for s in study['selected']).count('bonafide') == 10
    assert {g: sum(s['generator'] == g for s in study['selected']) for g in ('chatterbox', 'f5')} == {'chatterbox': 5, 'f5': 5}
    queue = store.queue('tester')
    assert len(queue) == 20 and {a['phase'] for a in queue} == {'challenge'}
    assert set(queue[0]) == {'id', 'phase', 'role', 'guideline', 'due_at', 'submitted_at', 'revealed_at', 'media_error', 'ordinal'}
    assert not {'reference', 'score', 'trial_id', 'attack', 'path'} & set(store.blind_review(queue[0]['id']))
    levels = []
    for a in queue:
        audio, rate = store.review_audio(a['id'])
        assert rate == 16000  # sample rate never reveals origin
        levels.append(20*np.log10(np.sqrt(np.mean(audio.astype(np.float64)**2))))
    assert np.allclose(levels, -26, atol=.1)  # loudness never reveals origin
    assert not review_rows(store)
    assert manifest_export(store)['selection'].startswith('Withheld')
    again = Store(tmp_path/'again.sqlite')
    try:
        assert [s['source_id'] for s in prepare_challenge(again, manifest, receipt, 'tester', root=tmp_path)['selected']] == \
               [s['source_id'] for s in study['selected']]
    finally:
        again.close()
    batch = progress(store, 'tester')['batches']
    assert [(b['phase'], b['label'], b['total']) for b in batch] == [('challenge', 'Listening challenge', 20)]


def test_refuses_mismatched_receipt_and_reuse(challenge, tmp_path):
    store, _, manifest, receipt = challenge
    with pytest.raises(ValueError, match='new database'):
        prepare_challenge(store, manifest, receipt, 'tester', root=tmp_path)
    other = Store(tmp_path/'other.sqlite')
    try:
        with pytest.raises(ValueError, match='exact frozen manifest'):
            prepare_challenge(other, manifest, {**receipt, 'manifest_sha256': 'different'}, 'tester', root=tmp_path)
    finally:
        other.close()


def test_reveal_scores_you_against_the_frozen_detector(challenge):
    store, study, _, receipt = challenge
    source = {s['trial_id']: s for s in study['selected']}
    trials = {r['id']: r['trial_id'] for r in store.rows('SELECT id,trial_id FROM recordings')}
    fake_seen = real_seen = 0
    expected = {'fooled': 0, 'caught_when_fooled': 0, 'reals_flagged': 0, 'false_alarms': 0}
    for a in store.queue('tester'):
        clip = source[trials[store.assignment(a['id'])['recording_id']]]
        flagged = clip['detector_score'] >= .5
        if clip['label'] == 'synthetic':
            judgment = ('human', 'human', 'uncertain')[fake_seen] if fake_seen < 3 else 'synthetic'
            fake_seen += 1
            if judgment != 'synthetic':
                expected['fooled'] += 1
                expected['caught_when_fooled'] += flagged
        else:
            judgment = 'synthetic' if real_seen == 0 else 'human'
            real_seen += 1
            expected['reals_flagged'] += judgment == 'synthetic'
            expected['false_alarms'] += flagged
        store.save(a['id'], payload(judgment=judgment, intelligibility='clear'), 'submitted')
    store.reveal_phase('challenge')
    result = challenge_summary(review_rows(store))
    assert result['clips'] == 20 and result['fakes'] == result['reals'] == 10
    assert result['fooled_you'] == expected['fooled'] == 3 and result['fooled_judged_human'] == 2
    assert result['detector_caught_when_fooled'] == expected['caught_when_fooled']
    assert result['reals_you_flagged'] == expected['reals_flagged'] == 1
    assert result['detector_false_alarms'] == expected['false_alarms']
    assert result['detector_caught'] == sum(s['detector_score'] >= .5 for s in study['selected'] if s['label'] == 'synthetic')
    rows = review_rows(store)
    assert all(r['prediction'] == ('synthetic' if r['score'] >= .5 else 'human') for r in rows)  # higher = synthetic here
    assert manifest_export(store)['selected'] == study['selected']


def test_challenge_screens_render_without_leaking(challenge, monkeypatch):
    from streamlit.testing.v1 import AppTest
    store = challenge[0]
    monkeypatch.setenv('AUDIO_LAB_DB', str(store.path))
    app = AppTest.from_file(APP, default_timeout=30).run()
    for name in ['Workspace', 'Review', 'Quality Review', 'Findings & Export']:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception, name
        text = ' '.join(str(e.value) for e in [*app.markdown, *app.caption, *app.metric] if hasattr(e, 'value'))
        assert 'CH_0' not in text and 'chatterbox' not in text.lower(), name


def test_cue_analysis_counts_against_the_reference():
    from lab.reporting import cue_analysis
    rows = [{'reference': 'synthetic', 'cues': ['flat or uniform emotion']},
            {'reference': 'synthetic', 'cues': ['flat or uniform emotion', 'too-clean studio sound']},
            {'reference': 'human', 'cues': ['too-clean studio sound', 'natural breaths or mouth sounds']},
            {'reference': 'human', 'cues': []}]
    result = {c['cue']: c for c in cue_analysis(rows)}
    assert result['flat or uniform emotion'] == {'cue': 'flat or uniform emotion', 'leans': 'synthetic', 'marked': 2,
                                                 'synthetic': 2, 'human': 0, 'synthetic_share': 1.0, 'base_rate': .5}
    assert result['too-clean studio sound']['synthetic_share'] == .5
    assert result['natural breaths or mouth sounds']['leans'] == 'human'
    assert cue_analysis([]) == []


def test_cues_are_validated(challenge):
    store = challenge[0]
    aid = store.queue('tester')[0]['id']
    for bad in (['not a cue'], ['flat or uniform emotion', 'flat or uniform emotion']):
        with pytest.raises(ValueError, match='cues'):
            store.save(aid, payload(cues=bad), 'draft')
    with pytest.raises(ValueError, match='other cue'):
        store.save(aid, payload(cues=['other cue'], rationale=''), 'submitted')
    store.save(aid, payload(cues=['metallic or robotic timbre', 'natural breaths or mouth sounds']), 'draft')
