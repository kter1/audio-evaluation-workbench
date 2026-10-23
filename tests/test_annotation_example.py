from copy import deepcopy
import pytest

from scripts.build_annotation_example import transcript_review


def test_transcript_timing_and_audio_identity_fail_closed():
    record = {'sample_sha256': 'one', 'words': [{'text': 'Hetty', 'start': 0, 'end': 1}]}
    with pytest.raises(ValueError, match='different recording'):
        transcript_review(record, 'two')
    for start, end in ((-1, 1), (1, .5), (0, 31), (float('nan'), 1), (None, 1)):
        broken = deepcopy(record)
        broken['words'][0].update(start=start, end=end)
        with pytest.raises(ValueError, match='timestamps'):
            transcript_review(broken, 'one')


def test_deletion_at_transcript_end_does_not_invent_timestamps():
    record = {'sample_sha256': 'one', 'words': [{'text': 'Hetty', 'start': 0, 'end': 1}]}
    differences, phrases = transcript_review(record, 'one')
    assert differences[-1]['model_text'] == '[omitted]'
    assert differences[-1]['start'] == 0 and differences[-1]['end'] == 1
    assert phrases == [{'start': 0, 'end': 1, 'text': 'Hetty'}]


def test_zero_duration_model_word_is_preserved_as_estimated_timing():
    record = {'sample_sha256': 'one', 'words': [
        {'text': 'Hetty', 'start': 0, 'end': .3},
        {'text': 'Sorrel', 'start': .3, 'end': .3},
    ]}
    _, phrases = transcript_review(record, 'one')
    assert phrases[0]['text'] == 'Hetty Sorrel'
    assert record['words'][1]['start'] == record['words'][1]['end']
