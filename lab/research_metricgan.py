"""Research-only full-length MetricGAN+ port; not a production detector input.

Architecture/inference adapted from SpeechBrain, Apache-2.0, Szu-Wei Fu and
SpeechBrain contributors. See lab/vendor/SPEECHBRAIN-LICENSE and pinned sources
in docs/metricgan-acquisition-v1.json. No SpeechBrain package or remote code load.
"""
from hashlib import sha1, sha256
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class MetricGANSession:
    def __init__(self, directory):
        spec=json.loads((ROOT/'docs/metricgan-acquisition-v1.json').read_text())
        self.directory=Path(directory)
        entry=next(f for f in spec['files'] if f['name']=='enhance_model.ckpt')
        path=self.directory/entry['name']
        if path.stat().st_size!=entry['size']:raise ValueError('MetricGAN checkpoint size mismatch')
        raw=path.read_bytes()
        if sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=entry['git_blob_sha1']:
            raise ValueError('MetricGAN checkpoint differs from pinned model')
        import torch
        from torch import nn
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.blstm=nn.LSTM(257,200,num_layers=2,bidirectional=True,batch_first=True)
                self.linear1=nn.Linear(400,300)
                self.linear2=nn.Linear(300,257)
                self.slope=nn.Parameter(torch.ones(257))
            def forward(self, features):
                x,_=self.blstm(features)
                x=self.linear2(torch.nn.functional.leaky_relu(self.linear1(x),negative_slope=.3))
                return 1.2*torch.sigmoid(self.slope*x)
        self.torch=torch;self.model=Generator()
        state=torch.load(path,map_location='cpu',weights_only=True)
        mapped={k.replace('blstm.rnn.','blstm.').replace('Learnable_sigmoid.slope','slope'):v for k,v in state.items()}
        if len(mapped)!=len(state):raise ValueError('Colliding checkpoint keys')
        self.model.load_state_dict(mapped,strict=True);self.model.eval()
        self.window=torch.hamming_window(512)
        self.manifest={'repository':spec['model_repository'],'revision':spec['model_revision'],
            'model_sha256':sha256(raw).hexdigest(),'source_revision':spec['source_revision'],
            'policy':'Full-length single-recording inference; log magnitude mask, noisy phase, upstream peak normalization; exact-zero input returned unchanged'}

    def predict(self, samples):
        x=np.asarray(samples,dtype=np.float32)
        if x.ndim!=1 or len(x)<8000 or not np.isfinite(x).all():raise ValueError('Expected finite mono16 >=0.5 seconds')
        if not np.any(x):return x.copy()
        torch=self.torch
        with torch.inference_mode():
            wav=torch.from_numpy(x)[None]
            spectrum=torch.stft(wav,n_fft=512,hop_length=256,win_length=512,window=self.window,
                                center=True,pad_mode='constant',normalized=False,onesided=True,return_complex=True)
            # Match upstream real/imag power summation and epsilon before square root.
            magnitude=torch.sqrt(spectrum.real.square()+spectrum.imag.square()+1e-14)
            features=torch.log1p(magnitude.transpose(1,2))
            mask=self.model(features)
            enhanced=torch.expm1(mask*features).transpose(1,2)
            phase=torch.atan2(spectrum.imag,spectrum.real)
            complex_output=torch.complex(enhanced*torch.cos(phase),enhanced*torch.sin(phase))
            y=torch.istft(complex_output,n_fft=512,hop_length=256,win_length=512,window=self.window,
                          center=True,onesided=True,length=len(x))
            y=y/(y.abs().amax(dim=1,keepdim=True)+1e-14)
        result=y[0].numpy()
        if len(result)!=len(x) or not np.isfinite(result).all():raise ValueError('Invalid MetricGAN output')
        return result
