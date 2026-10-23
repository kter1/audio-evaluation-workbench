"""Compare research port with inspected, pinned upstream tensor implementations.

Executes only named AST definitions from locally verified official source.
SpeechBrain's full-length LSTM wrapper is represented by a plain torch LSTM;
this is not a claim of full installed-SpeechBrain package equivalence.
"""
import argparse
import ast
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.research_metricgan import MetricGANSession


def definitions(path,names,namespace,expected):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected:raise ValueError('Upstream source differs from acquisition receipt')
    tree=ast.parse(raw.decode())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in selected}!=set(names):raise ValueError('Reference definitions missing')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from torch import nn
    torch.set_num_threads(1)
    model=MetricGANSession(args.directory)
    receipt=json.loads((args.directory/'acquisition.json').read_text())
    hashes={Path(r['path']).name:r['sha256'] for r in receipt['files']}
    class FullLengthLSTM(nn.Module):
        def __init__(self,**kwargs):
            super().__init__();self.rnn=nn.LSTM(**kwargs,batch_first=True)
        def forward(self,x,lengths=None):return self.rnn(x)
    ns={'torch':torch,'nn':nn,'FilterProperties':object,'spectral_norm':nn.utils.spectral_norm,
        'sb':SimpleNamespace(nnet=SimpleNamespace(RNN=SimpleNamespace(LSTM=FullLengthLSTM)))}
    source=args.directory/'upstream'
    definitions(source/'MetricGAN.py',['xavier_init_layer','Learnable_sigmoid','EnhancementGenerator'],ns,hashes['MetricGAN.py'])
    definitions(source/'features.py',['STFT','ISTFT','spectral_magnitude'],ns,hashes['features.py'])
    definitions(source/'signal_processing.py',['compute_amplitude','normalize','resynthesize'],ns,hashes['signal_processing.py'])
    reference=ns['EnhancementGenerator']();reference.load_state_dict(torch.load(args.directory/'enhance_model.ckpt',map_location='cpu',weights_only=True));reference.eval()
    stft=ns['STFT'](sample_rate=16000,win_length=32,hop_length=16,n_fft=512,window_fn=torch.hamming_window)
    istft=ns['ISTFT'](sample_rate=16000,win_length=32,hop_length=16,n_fft=512,window_fn=torch.hamming_window)
    rows=[]
    for count in (8000,16001,64000):
        for kind in ('noise','tone'):
            x=(np.random.default_rng(count).normal(0,.1,count) if kind=='noise' else .1*np.sin(2*np.pi*440*np.arange(count)/16000)).astype('float32')
            with torch.inference_mode():
                wav=torch.from_numpy(x)[None]
                features=torch.log1p(ns['spectral_magnitude'](stft(wav),power=.5))
                mask=reference(features,lengths=torch.ones(1))
                y=ns['resynthesize'](torch.expm1(mask*features),wav,stft,istft)[0].numpy()
            actual=model.predict(x)
            error=float(np.max(np.abs(actual-y)))
            if not np.allclose(actual,y,rtol=1e-5,atol=1e-6):raise ValueError('Port differs from pinned reference')
            rows.append({'samples':count,'kind':kind,'maximum_absolute_error':error})
    result={'schema_version':'metricgan-port-validation-1.0','manifest':model.manifest,'cases':rows,
        'max_absolute_error':max(r['maximum_absolute_error'] for r in rows),
        'scope':'Six full-length tensor parity fixtures against pinned upstream definitions with plain full-length torch LSTM wrapper. Not installed-package parity or speech enhancement accuracy. Exact-zero guard intentionally differs from upstream epsilon/peak-normalization behavior.',
        'code_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in ('lab/research_metricgan.py','scripts/validate_metricgan_port.py')}}
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print('6/6 parity cases passed; max absolute error',result['max_absolute_error'])


if __name__=='__main__':main()
