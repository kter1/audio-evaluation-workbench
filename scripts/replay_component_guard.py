"""Replay the component-disagreement guard on already-consumed development scores."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.core_screen import prepare_case, summarize
from lab.forensics import RATE, POLICY, abstention_band, decision, diagnostics
from lab.vad_stress import transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--acquisition',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-product',action='store_true',help='Run full local product on the known wrong definitive cases only')
    args=parser.parse_args()
    raw=args.screen.read_bytes(); original=json.loads(raw)
    acquisition_raw=args.acquisition.read_bytes(); acquisition=json.loads(acquisition_raw)
    if original['acquisition_sha256']!=sha256(acquisition_raw).hexdigest():
        raise ValueError('Acquisition differs from original screen')
    spec=json.loads((ROOT/'lab/fusion-v4.json').read_text())
    if spec['threshold']!=original['threshold'] or original['model']['fusion_sha256']!=sha256((ROOT/'lab/fusion-v4.json').read_bytes()).hexdigest():
        raise ValueError('Frozen model/threshold changed')
    files={'voxenes-2026:'+r['name']:r for r in acquisition['files']}
    rows=[]; followups=[]
    for previous in original['results']:
        base_id=previous['id'].rsplit('::',1)[0]; item=files[base_id]
        x,*_=prepare_case(item,ROOT)
        condition=previous['condition']
        if condition!='original':
            seed=int(sha256((item['sha256']+':core-screen-1.0:'+condition).encode()).hexdigest()[:16],16)
            x,_,_=transform(x,None,condition,seed)
        if sha256(x.astype('<f4').tobytes()).hexdigest()!=previous['transformed_samples_sha256']:
            raise ValueError('Transformed input does not match prior screen')
        _,metrics,_=diagnostics(x[:,None],RATE)
        score={'start':0.,'end':len(x)/RATE,'synthetic_score':previous['synthetic_score'],'components':previous['components']}
        current=decision([score],metrics,len(x)/RATE,spec['threshold'],abstention_band(spec['threshold'],2.))
        rows.append({**previous,'previous_assessment':previous['assessment'],'assessment':current})
        was_wrong=(previous['label']=='synthetic' and previous['assessment']['assessment']=='Human speech indicated') or (previous['label']=='human' and previous['assessment']['assessment']=='Synthetic speech indicated')
        if args.verify_product and was_wrong:
            from signaltrace_server import infer_bytes
            buffer=BytesIO();sf.write(buffer,x,RATE,format='WAV',subtype='FLOAT')
            actual=infer_bytes(buffer.getvalue())
            followups.append({'id':previous['id'],'model_executed':actual['model_executed'],
                'assessment':actual['assessment'],'links':actual['links']})
            if actual['assessment']['assessment']!='Inconclusive':
                raise ValueError('Known contradictory product case was not made inconclusive')
    conditions=original['by_condition'].keys()
    result={'schema_version':'component-guard-replay-1.0','policy':POLICY,
        'original_screen_sha256':sha256(raw).hexdigest(),'original_clip_count':original['original_clip_count'],
        'case_count':len(rows),'source_groups':original['source_groups'],
        'before':original['by_condition'],
        'after':{condition:summarize([r for r in rows if r['condition']==condition],spec['threshold']) for condition in conditions},
        'results':rows,'targeted_product_followups':followups,
        'code_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/replay_component_guard.py','lab/forensics.py','signaltrace_server.py','lab/core_screen.py','lab/vad_stress.py','lab/vad_channel_stress.py')},
        'scope':'Post-hoc development guard selected after observing the two noisy Qwen3 errors. Same scores/thresholds; no retraining. '
                'Not independent validation, improved raw detection or proof of world-class performance. '
                'Full-product reruns target known errors only; all other rows replay core score/quality/uncertainty checks.'}
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for condition in conditions:
        before=sum(v['wrong_definitive'] for v in result['before'][condition].values())
        after=sum(v['wrong_definitive'] for v in result['after'][condition].values())
        uncertain=sum(v['inconclusive'] for v in result['after'][condition].values())
        print(condition,'wrong definitive',before,'->',after,'inconclusive',uncertain)


if __name__=='__main__':
    main()
