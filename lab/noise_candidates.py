"""Fixed development-only preprocessing candidates; neither is a production denoiser."""
import numpy as np
from scipy import signal

RATE = 16000
CONFIG = {
    'spectral_floor': {'nperseg':512, 'hop':128, 'noise_quantile':.2, 'minimum_gain':.2,
                       'noise_assumption':'Per-frequency lower temporal quantile; continuous speech can be removed as noise'},
    'lowpass_control': {'cutoff_hz':3400, 'butterworth_order':6, 'phase':'zero phase'},
}


def preprocess(samples, candidate):
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim!=1 or len(x)<8000 or not np.isfinite(x).all():
        raise ValueError('Expected at least 0.5 seconds finite mono16 samples')
    if candidate=='lowpass_control':
        result=signal.sosfiltfilt(signal.butter(6,3400,fs=RATE,output='sos'),x)
    elif candidate=='spectral_floor':
        _,_,spectrum=signal.stft(x,fs=RATE,window='hann',nperseg=512,noverlap=384,boundary='zeros',padded=True)
        power=np.abs(spectrum)**2
        noise=np.quantile(power,.2,axis=1,keepdims=True)
        gain=np.maximum(.2,np.sqrt(np.maximum(0.,1-noise/np.maximum(power,1e-20))))
        _,result=signal.istft(spectrum*gain,fs=RATE,window='hann',nperseg=512,noverlap=384,boundary=True)
        result=result[:len(x)]
    else:
        raise ValueError('Unknown fixed preprocessing candidate')
    if len(result)!=len(x) or not np.isfinite(result).all():
        raise ValueError('Candidate changed timing or produced nonfinite audio')
    return np.asarray(result,dtype=np.float32)
