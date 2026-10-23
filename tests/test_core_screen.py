from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from lab.core_screen import prepare_case, screen, summarize


def case(tmp_path):
    x = np.linspace(-.1,.1,6*16000,dtype='float32')
    path = tmp_path/'voice.wav'; sf.write(path,x,16000,subtype='FLOAT')
    return {'path':path.name,'name':'publisher/voice.wav','sha256':sha256(path.read_bytes()).hexdigest(),
            'label':'synthetic','group':'fixture'}


def test_center_crop_is_fixed_and_audio_hash_checked(tmp_path):
    row = case(tmp_path)
    x, _, _, p = prepare_case(row,tmp_path)
    assert len(x)==64000 and p['crop_start_sample_16k']==16000
    raw, _ = sf.read(tmp_path/row['path'],dtype='float32')
    assert np.array_equal(x,raw[16000:80000])
    row['sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):prepare_case(row,tmp_path)


def test_summary_keeps_raw_and_definitive_errors_separate():
    rows=[{'group':'human','label':'human','synthetic_score':.9,'assessment':{'assessment':'Inconclusive'}},
          {'group':'fake','label':'synthetic','synthetic_score':.01,'assessment':{'assessment':'Human speech indicated'}}]
    result=summarize(rows,.2)
    assert result['human']['raw_false_positive']==1 and result['human']['wrong_definitive']==0
    assert result['fake']['raw_synthetic_detected']==0 and result['fake']['wrong_definitive']==1
    with pytest.raises(ValueError):summarize([{**rows[0],'synthetic_score':float('nan')}],.2)


def test_screen_marks_all_cases_consumed_and_does_not_fit(tmp_path):
    row=case(tmp_path)
    class Detector:
        threshold=.2
        band_logit=2
        manifest={'repository':'fixture'}
        def score(self,x):
            return {'synthetic_score':.99,'human_score':.01,'logits':[0.,4.6], 'components':{'fixture':.99},
                    'model_input_sha256':'a'*64}
    result=screen({'files':[row],'spec':{'selection':'fixture only'}},tmp_path,Detector())
    assert result['source_groups']==1
    assert result['by_group']['fixture']['raw_synthetic_detected']==1
    assert result['input_identities'][0]['record_id'].startswith('voxenes-2026:')
    assert 'audio:'+row['sha256'] in result['input_identities'][0]['identity_tokens']
    assert result['results'][0]['model_input_sha256'] == 'a'*64


def test_stress_reports_each_condition_without_inflating_source_support(tmp_path):
    row=case(tmp_path)
    class Detector:
        threshold=.2
        manifest={}
        def score(self,x):return {'synthetic_score':.9,'human_score':.1,'logits':[0.,2.2]}
    result=screen({'files':[row],'spec':{'selection':'fixture'}},tmp_path,Detector(),stress=True)
    assert result['case_count']==5 and result['original_clip_count']==1
    assert result['source_groups']==1
    assert set(result['by_condition'])=={'original','gain_minus12db','added_noise20db','bandwidth8k','vorbis_roundtrip'}
    assert all(groups['fixture']['clips']==1 for groups in result['by_condition'].values())
    assert len({row['id'] for row in result['results']})==5
