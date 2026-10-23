from copy import deepcopy
import random

import pytest

from scripts.expand_asvspoof5_corpus import reserve
from scripts.audit_asvspoof5_expansion import check_reservations


def sources():
    rows=[]
    for i in range(700):
        group=f'speaker-{i:04d}'
        rows.append({'id':group+'-human','speaker':group,'label':'human'})
        if i < 350:
            rows.append({'id':group+'-synthetic','speaker':group,'label':'synthetic'})
    pilot=[]
    for i in range(20):
        group=f'speaker-{i:04d}'
        label='human' if i%2 else 'synthetic'
        pilot.append({'id':group+'-'+label,'speaker':group,'label':label,
                      'partition':'calibration_reserved' if i<10 else 'evaluation_reserved'})
    return rows,pilot


def test_expansion_preserves_reservations_and_all_class_group_minimums():
    entries,pilot=sources()
    selected=reserve(entries,pilot)
    expected=deepcopy(selected)
    random.Random(22).shuffle(entries)
    assert reserve(entries,pilot)==expected
    roles={r['speaker']:r['partition'] for r in selected}
    assert all(roles[r['speaker']]==r['partition'] for r in pilot)
    assert {r['id'] for r in pilot} <= {r['id'] for r in selected}
    for partition,minimum_synthetic in [('calibration_reserved',100),('evaluation_reserved',200)]:
        part=[r for r in selected if r['partition']==partition]
        assert len({r['speaker'] for r in part if r['label']=='human'})>=300
        assert len({r['speaker'] for r in part if r['label']=='synthetic'})>=minimum_synthetic
    calibration={r['speaker'] for r in selected if r['partition']=='calibration_reserved'}
    evaluation={r['speaker'] for r in selected if r['partition']=='evaluation_reserved'}
    assert not calibration & evaluation


def test_insufficient_sources_or_duplicate_ids_fail_without_relaxing_targets():
    entries,pilot=sources()
    with pytest.raises(ValueError,match='Duplicate'):
        reserve(entries+[entries[0]],pilot)
    with pytest.raises(ValueError,match='minimum unavailable'):
        reserve(entries[:30],pilot)


def audit_fixture():
    entries, pilot = sources()
    for row in entries:
        row['sha256'] = 'digest-'+row['id']
    for row in pilot:
        row['source_sha256'] = 'digest-'+row['id']
    return reserve(entries, pilot), pilot


def test_independent_audit_rejects_leakage_duplicates_and_changed_pilot():
    selected, pilot = audit_fixture()
    assert check_reservations(selected, pilot)['evaluation_reserved']['human'] >= 300
    crossed = deepcopy(selected)
    speaker = next(r['speaker'] for r in crossed if sum(x['speaker']==r['speaker'] for x in crossed)==2)
    row = next(r for r in crossed if r['speaker']==speaker)
    row['partition'] = ('evaluation_reserved' if row['partition']=='calibration_reserved' else 'calibration_reserved')
    with pytest.raises(ValueError, match='crosses'):
        check_reservations(crossed, pilot)
    with pytest.raises(ValueError, match='Duplicate'):
        check_reservations(selected+[selected[0]], pilot)
    changed = deepcopy(selected)
    next(r for r in changed if r['id']==pilot[0]['id'])['sha256']='changed'
    with pytest.raises(ValueError, match='bytes changed'):
        check_reservations(changed, pilot)


def test_independent_audit_rejects_underfilled_partitions_and_invalid_labels():
    selected, pilot = audit_fixture()
    with pytest.raises(ValueError, match='minimum unmet'):
        check_reservations(selected[:10], [])
    selected[0]['label']='unknown'
    with pytest.raises(ValueError, match='Unknown'):
        check_reservations(selected, pilot)
