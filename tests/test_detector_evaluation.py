import numpy as np
import pytest
from lab.aasist import AASISTDetector, repeat_pad
from lab.evaluation import summarize, wilson
from lab.learning import forensic_examples
from scripts.prepare_evaluation import choose_candidates


def test_restricted_repeat_padding_and_order():
    x=np.array([1,2,3],dtype=np.float32)
    assert list(repeat_pad(x,8))==[1,2,3,1,2,3,1,2]
    with pytest.raises(ValueError):repeat_pad(np.zeros(64601))
    with pytest.raises(ValueError):repeat_pad(np.array([]))


@pytest.mark.parametrize('n',[8000,40000,64600,64601,64640,66658,96000,129201,1920000])
def test_aasist_tail_has_full_context_and_covers_audio(n):
    # Window planning needs no weights or torch execution.
    detector=object.__new__(AASISTDetector)
    spans=detector.spans(np.zeros(n))
    assert spans[0][0]==0 and spans[-1][1]==n
    assert len({a for a,b in spans})==len(spans)
    assert all(b-a==min(n,64600) for a,b in spans)
    assert all(spans[i][1]>=spans[i+1][0] for i in range(len(spans)-1))


def test_evaluation_metrics_and_missing_class():
    rows=[{'label':'human','score':.1,'assessment':'Human speech indicated'},
          {'label':'human','score':.8,'assessment':'Synthetic speech indicated'},
          {'label':'synthetic','score':.8,'assessment':'Synthetic speech indicated'},
          {'label':'synthetic','score':.4,'assessment':'Inconclusive'}]
    r=summarize(rows)
    assert r['correct']==2 and r['false_positive']==r['false_negative']==1
    assert r['balanced_accuracy']==.5 and r['accuracy_among_decided']==pytest.approx(2/3)
    assert r['roc_auc']==pytest.approx(.625)
    assert r['abstained']==1 and r['decided']==3
    assert wilson(38,40)==pytest.approx([.8349612263085903,.9861793326138516])
    assert summarize(rows[:2])['roc_auc'] is None
    assert summarize(rows[:2])['balanced_accuracy'] is None
    assert wilson(0,0) is None


def test_candidate_selection_is_fixed_balanced_and_source_unique():
    rows=[{'trial_id':f'r{i:03}','source_group':f'g{i//2}','label':'human' if i<40 else 'synthetic'} for i in range(80)]
    a=choose_candidates(rows,seed=42,per_class=10)
    assert a==choose_candidates(list(reversed(rows)),seed=42,per_class=10)
    assert len(a)==len({r['source_group'] for r in a})==20
    assert sum(r['label']=='human' for r in a)==10


def test_forensic_examples_exclude_short_and_assigned_sources(db):
    safe=db.one("SELECT id FROM recordings WHERE subset='progress' LIMIT 1")['id']
    assigned=db.one('SELECT recording_id FROM assignments LIMIT 1')['recording_id']
    with db.conn:
        db.set_setting('forensic_examples',[assigned,safe])
        db.conn.execute('UPDATE recordings SET duration=3 WHERE id IN (?,?)',(safe,assigned))
    assert [r['id'] for r in forensic_examples(db)]==[safe]
    with db.conn:db.conn.execute('UPDATE recordings SET duration=2.05 WHERE id=?',(safe,))
    assert not forensic_examples(db)


def test_aasist_native_spoof_logit_is_exported_as_synthetic():
    import torch
    model=object.__new__(AASISTDetector)
    model.torch=torch
    model.model=lambda x,Freq_aug=False:(None,torch.tensor([[5.,-5.]]))
    scored=model.score(np.zeros(64600,dtype=np.float32))
    assert scored['logits']==[-5.,5.]
    assert scored['synthetic_score']>.999
    assert scored['human_score']<.001
