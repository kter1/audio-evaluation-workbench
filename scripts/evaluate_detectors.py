"""Frozen two-stage model selection, then an untouched local test subset."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.forensics import Detector, POLICY, analyze, save_report
from lab.aasist import AASISTDetector
from lab.evaluation import summarize
from lab.store import digest


def score_cases(model, cases):
    rows=[]
    for i,case in enumerate(cases,1):
        blob=Path(case['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=case['sha256']:raise ValueError('Audio checksum changed')
        report,_=analyze(blob,model,robustness=False)
        save_report(report,Path('data/evaluation/runs'))
        rows.append({k:case[k] for k in ('id','trial_id','source_group','label','codec','attack','duration','sha256')} |
                    {'score':report['assessment']['synthetic_score'],'assessment':report['assessment']['assessment'],
                     'reasons':report['assessment']['abstention_reasons']})
        if i%10==0:print(model.name,i,'/',len(cases),flush=True)
    return rows


def main():
    manifest=json.loads(Path('data/evaluation/manifest.json').read_text())
    body={k:v for k,v in manifest.items() if k!='sha256'}
    if digest(body)!=manifest['sha256']:raise ValueError('Evaluation manifest changed')
    selection=[r for r in manifest['cases'] if r['split']=='selection']
    test=[r for r in manifest['cases'] if r['split']=='test']
    assert len(selection)==len(test)==40
    assert len({r['source_group'] for r in selection+test})==80
    models={'wav2vec2':Detector(),'aasist':AASISTDetector()}
    result={'version':manifest['version'],'created_at':datetime.now(timezone.utc).isoformat(), 'manifest_sha256':manifest['sha256'],
            'selection_rule':manifest['model_selection_rule'],'policy':POLICY,
            'models':{k:{**v.manifest,'parameters':v.parameters,'name':v.name} for k,v in models.items()},
            'selection':{},'test':{},'demo_ids':manifest['demo_ids']}
    for key,model in models.items():
        rows=score_cases(model,selection)
        result['selection'][key]={'summary':summarize(rows),'rows':rows}
        print('SELECTION',key,json.dumps(result['selection'][key]['summary']),flush=True)
    chosen=max(models,key=lambda k:result['selection'][k]['summary']['balanced_accuracy'])
    result['selected_model']=chosen
    # Written before any test inference. A rerun must preserve this frozen choice.
    frozen_path=Path('data/evaluation')/('selection-'+manifest['sha256']+'.json')
    selection_record={'manifest_sha256':manifest['sha256'],'selected_model':chosen,'rule':manifest['model_selection_rule']}
    if frozen_path.exists() and json.loads(frozen_path.read_text())!=selection_record:raise ValueError('Frozen model selection conflict')
    frozen_path.write_text(json.dumps(selection_record,indent=2)+'\n')
    print('FROZEN CHOICE:',chosen,flush=True)
    rows=score_cases(models[chosen],test)
    result['test']={'summary':summarize(rows),'rows':rows}
    result['limitations']=['40 local test cases; uncertainty is large. Wilson interval assumes independent cases; speaker and codec correlations may remain.',
        'Sources are separated between selection, test, progress, and the annotation study. Complete training overlap has not been audited.',
        'ASVspoof 2021 LA includes 2019-era spoof attacks and channel conditions. These results do not establish performance on modern generators or arbitrary uploads.',
        'No thresholds, preprocessing, or abstention rules were fitted to the test set. Only the selected model was tested on it.']
    Path('docs/detector-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TEST',chosen,json.dumps(result['test']['summary']),flush=True)


if __name__=='__main__':main()
