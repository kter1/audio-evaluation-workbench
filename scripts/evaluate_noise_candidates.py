"""Compare two fixed preprocessing candidates against unchanged core detection."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core_screen import prepare_case, summarize
from lab.forensics import POLICY, RATE, abstention_band, decision, diagnostics
from lab.noise_candidates import CONFIG, preprocess
from lab.vad_stress import transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metricgan-directory',type=Path,help='Run only unchanged + pinned neural enhancement as a separate third candidate')
    args=parser.parse_args()
    raw=args.acquisition.read_bytes();acquisition=json.loads(raw)
    baseline_raw=args.baseline.read_bytes();baseline=json.loads(baseline_raw)
    if baseline['acquisition_sha256']!=sha256(raw).hexdigest():raise ValueError('Baseline acquisition mismatch')
    if len(acquisition['files'])!=45:raise ValueError('Fixed 45-clip development cohort required')
    from lab.fusion import FusionDetector
    import torch
    started=time.perf_counter();detector=FusionDetector();torch.set_num_threads(1)
    if detector.manifest!=baseline['model'] or detector.threshold!=baseline['threshold']:
        raise ValueError('Baseline model or threshold mismatch')
    enhancer=None
    config=CONFIG
    if args.metricgan_directory:
        from lab.research_metricgan import MetricGANSession
        enhancer=MetricGANSession(args.metricgan_directory)
        config={'metricgan':enhancer.manifest}
    indexed={r['id']:r for r in baseline['results']}
    if len(indexed)!=len(baseline['results']):raise ValueError('Duplicate baseline IDs')
    results=[];identities=[]
    for n,item in enumerate(acquisition['files']):
        original,*_=prepare_case(item,ROOT)
        for condition in ('original','added_noise20db'):
            record_id='voxenes-2026:'+item['name']+'::'+condition
            previous=indexed[record_id]
            x=original
            if condition!='original':
                seed=int(sha256((item['sha256']+':core-screen-1.0:'+condition).encode()).hexdigest()[:16],16)
                x,_,_=transform(x,None,condition,seed)
            if sha256(x.astype('<f4').tobytes()).hexdigest()!=previous['transformed_samples_sha256']:
                raise ValueError('Reconstructed baseline audio differs')
            for candidate in ('unchanged',*config):
                y=x if candidate=='unchanged' else enhancer.predict(x) if candidate=='metricgan' else preprocess(x,candidate)
                scores=({'synthetic_score':previous['synthetic_score'],'components':previous['components']}
                        if candidate=='unchanged' else detector.score(y))
                _,metrics,warnings=diagnostics(y[:,None],RATE)
                assessment=decision([{'start':0.,'end':len(y)/RATE,**scores}],metrics,len(y)/RATE,
                                    detector.threshold,abstention_band(detector.threshold,detector.band_logit))
                pcm_hash=sha256(y.astype('<f4').tobytes()).hexdigest()
                results.append({'id':record_id+'::'+candidate,'label':item['label'],'group':item['group'],
                    'candidate':candidate,'condition':condition,'synthetic_score':scores['synthetic_score'],
                    'components':scores['components'],'assessment':assessment,'warnings':warnings,
                    'analysis_samples_sha256':pcm_hash,'source_sha256':item['sha256']})
                identities.append({'record_id':record_id+'::'+candidate,'identity_tokens':[
                    'audio:'+item['sha256'],'analysis-pcm:'+pcm_hash,'voxenes-2026:unresolved-source-speaker-identities']})
        print('Scored source',n+1,'/45',flush=True)
    summaries={condition:{candidate:summarize([r for r in results if r['condition']==condition and r['candidate']==candidate],detector.threshold)
                         for candidate in ('unchanged',*config)} for condition in ('original','added_noise20db')}
    result={'schema_version':'core-preprocessing-experiment-1.0','mode':'supplied development authenticity labels',
        'config':config,'policy':POLICY,'threshold':detector.threshold,'model':detector.manifest,
        'acquisition_sha256':sha256(raw).hexdigest(),'baseline_sha256':sha256(baseline_raw).hexdigest(),
        'original_clip_count':45,'source_groups':1,'results':results,'summaries':summaries,'input_identities':identities,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads()},
        'runtime_including_load_seconds':time.perf_counter()-started,
        'code_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/evaluate_noise_candidates.py','lab/noise_candidates.py','lab/forensics.py','lab/core_screen.py','lab/fusion.py','lab/vad_stress.py','lab/research_metricgan.py')},
        'scope':'Fixed, development-only paired preprocessing experiment, including human controls in clean and added-noise conditions. '
                'No threshold tuning or classifier retraining. Unchanged scores reused; candidate scores freshly inferred. '
                'Inherited labels do not verify transformed audibility. Repeated variants are not independent samples. '
                'Not production promotion or independent validation; unknown training/source overlap.'}
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    directory=ROOT/'data/evaluation/method-runs';directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/(sha256(text.rstrip('\n').encode()).hexdigest()+'.json')
    with receipt.open('x') as stream:stream.write(text)
    with args.output.open('x') as stream:stream.write(text)
    for condition,methods in summaries.items():
        for name,groups in methods.items():
            print(condition,name,'human FP',sum(r['raw_false_positive'] for r in groups.values()),
                  'AI detected',sum(r['raw_synthetic_detected'] for r in groups.values()),
                  'AI indicated',sum(r['synthetic_indicated'] for r in groups.values()),
                  'inconclusive',sum(r['inconclusive'] for r in groups.values()))


if __name__=='__main__':main()
