"""Freeze v2: 40 selection, 40 retired regression, 40 fresh test cases."""
import json
from pathlib import Path
import random
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.forensics import read_audio
from lab.store import Store, digest


def choose_candidates(rows, seed=20261020, per_class=80):
    rows = sorted(rows, key=lambda r: r['trial_id'])
    random.Random(seed).shuffle(rows)
    selected, groups, counts = [], set(), {'human': 0, 'synthetic': 0}
    for row in rows:
        if row['source_group'] in groups or counts[row['label']] >= per_class:
            continue
        selected.append(row)
        groups.add(row['source_group'])
        counts[row['label']] += 1
        if len(selected) == 2*per_class:
            break
    return selected


def main():
    root = Path('data/evaluation'); root.mkdir(parents=True, exist_ok=True)
    output = root/'manifest.json'
    if output.exists():
        print('Frozen manifest already exists:', output); return
    db = Store('data/study.sqlite')
    rows = db.rows("""SELECT r.*,l.label FROM recordings r JOIN reference_labels l ON l.recording_id=r.id
      WHERE r.subset='eval' AND r.source_group IS NOT NULL
      AND r.source_group NOT IN (SELECT source_group FROM recordings WHERE subset='progress' AND source_group IS NOT NULL)
      AND r.source_group NOT IN (SELECT source_group FROM recordings WHERE id IN (SELECT recording_id FROM assignments))
      ORDER BY r.trial_id""")
    candidates = choose_candidates(rows)
    by_name = {r['trial_id']+'.flac': r for r in candidates}
    print('Extracting',len(candidates),'candidates, without using detector scores.',flush=True)
    with tarfile.open('data/asvspoof/ASVspoof2021_LA_eval.tar.gz', mode='r|gz') as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in by_name: continue
            if not member.isfile() or member.size > 20_000_000: raise ValueError('Unexpected archive entry')
            path = Path(by_name[name]['path']); path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): path.write_bytes(tar.extractfile(member).read())
    selected, retired, excluded, counts = [], [], [], {'human':0,'synthetic':0}
    for r in candidates:
        if counts[r['label']] >= 60: continue
        try:
            _,_,info = read_audio(Path(r['path']).read_bytes())
        except (ValueError,OSError) as error:
            excluded.append({'trial_id':r['trial_id'],'reason':str(error)}); continue
        phase = 'selection' if counts[r['label']] < 20 else ('retired' if counts[r['label']] < 40 else 'test')
        selected.append({k:r[k] for k in ('id','trial_id','source_group','label','codec','attack','path')} |
                        {'split':phase,'duration':info['duration_seconds'],'sha256':info['sha256']})
        counts[r['label']] += 1
    if counts != {'human':60,'synthetic':60}: raise ValueError('Insufficient decodable candidates')
    retired = [r for r in selected if r['split']=='retired']
    selected = [r for r in selected if r['split']!='retired']
    demos=[]
    for label in ('human','synthetic'):
        demos.extend(r['id'] for r in [r for r in selected if r['label']==label and 2.5<=r['duration']<=13][:2])
    body={'version':'2.0','seed':20261020,'selection_rule':'First 60 decodable cases per class after seeded permutation; 20/class selection, 20/class retired regression set, 20/class fresh test. No scores used. Unique source groups, excluding all progress sources and all assigned study sources.',
          'model_selection_rule':'Choose highest raw binary balanced accuracy on selection; tie favors existing wav2vec2. Freeze choice before evaluating test. Threshold 0.5, no tuning.',
          'models':['wav2vec2','aasist'],'cases':selected,'retired_cases':retired,'excluded':excluded,'demo_ids':demos,
          'demo_selection_rule':'First two cases per class in frozen order with duration 2.5–13 seconds; no score filter.',
          'archive_provenance':db.setting('audio_archive_provenance')}
    body['sha256']=digest(body)
    output.write_text(json.dumps(body,indent=2)+'\n')
    with db.conn:
        db.set_setting('forensic_examples',demos)
        for r in selected:
            db.conn.execute('UPDATE recordings SET duration=?,sha256=? WHERE id=?',(r['duration'],r['sha256'],r['id']))
    print('Frozen:',output,'cases:',len(selected),'excluded:',len(excluded),'SHA256:',body['sha256'],flush=True)
    db.close()


if __name__=='__main__':main()
