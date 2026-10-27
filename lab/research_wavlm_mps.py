"""Explicit MPS research acceleration, separate from the frozen CPU adapter."""
from hashlib import sha256
import time

import numpy as np

from lab.research_wavlm_aasist import WavLMAASISTDetector, preprocess


class WavLMMPSDetector(WavLMAASISTDetector):
    def __init__(self, directory=None):
        import torch
        if not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; use the CPU research adapter')
        super().__init__(directory)
        self.model.to('mps')
        torch.mps.synchronize()
        self.manifest = {**self.manifest, 'device': 'mps',
                         'torch_version': torch.__version__,
                         'precision': 'float32; no autocast or quantization',
                         'device_parity_required': True}

    def score(self, samples):
        started = time.perf_counter()
        waveform = preprocess(samples)
        input_hash = sha256(waveform.numpy().astype('<f4').tobytes()).hexdigest()
        with self.torch.inference_mode():
            native = self.model(waveform.to('mps'))[0]
            # Synchronize before copying output and recording elapsed work.
            self.torch.mps.synchronize()
            native_cpu = native.cpu()
            real_score = float(self.torch.sigmoid(native_cpu))
        native_logit = float(native_cpu)
        score = 1.-real_score
        if not np.isfinite(native_logit) or not 0 <= score <= 1:
            raise ValueError('MPS WavLM/AASIST returned an invalid score')
        self.timings.append({'seconds': time.perf_counter()-started,
                             'input_seconds': len(samples)/16000, 'model_seconds': 5.})
        return {'synthetic_score': score, 'human_score': real_score,
                'logits': [0., -native_logit], 'native_real_logit': native_logit,
                'model_input_sha256': input_hash}
