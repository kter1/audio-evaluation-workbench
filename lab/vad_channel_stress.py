"""Constructed transport/channel stress; no claim of real-device equivalence."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly
import soundfile as sf

RATE = 16000
CHANNEL_CONDITIONS = ('original', 'bandwidth8k', 'stereo_one_empty', 'stereo_antiphase',
                      'vorbis_roundtrip', 'crop_center500ms', 'crop_center1s', 'crop_center2s')


def crop_labels(intervals, start_sample, end_sample):
    if intervals is None:
        return None
    start, end = start_sample/RATE, end_sample/RATE
    duration = (end_sample-start_sample)/RATE
    return [{'start': max(0., i['start']-start), 'end': min(duration, i['end']-start)}
            for i in intervals if i['end'] > start and i['start'] < end]


def transform_channel(samples, intervals, condition, seed):
    x = np.asarray(samples, dtype=np.float32)
    if condition not in CHANNEL_CONDITIONS or condition == 'original':
        raise ValueError('Unknown channel/codec stress condition')
    labels = deepcopy(intervals)
    metadata = {'condition': condition, 'seed': seed, 'input_samples': len(x),
        'label_policy': 'Inherited source speech timing; transformed audibility not independently verified'}
    if condition == 'bandwidth8k':
        y = resample_poly(resample_poly(x, 1, 2), 2, 1)[:len(x)].astype(np.float32)
        metadata.update(intermediate_rate_hz=8000, method='scipy resample_poly default filter; down/up; final sample count preserved',
                        limitation='Constructed bandwidth restriction, not a measured telephone/device channel; filter edge effects possible')
    elif condition == 'stereo_one_empty':
        y = np.column_stack((x, np.zeros_like(x)))
        metadata.update(channels=2, expected_mono_gain_db=float(20*np.log10(.5)))
    elif condition == 'stereo_antiphase':
        y = np.column_stack((x, -x))
        metadata.update(channels=2, expected_mono='exact cancellation',
                        interpretation='Source channels retain speech; mono input loses it. A transport failure diagnostic, not standalone VAD accuracy.')
    elif condition.startswith('crop_center'):
        requested = {'crop_center500ms': 8000, 'crop_center1s': 16000, 'crop_center2s': 32000}[condition]
        count = min(len(x), requested)
        start = (len(x)-count)//2
        y = x[start:start+count].copy()
        labels = crop_labels(intervals, start, start+count)
        metadata.update(crop_start_sample=start, crop_end_sample=start+count, requested_samples=requested,
                        selection='Center by sample count, independent of labels and predictions')
    else:
        payload = BytesIO()
        sf.write(payload, x, RATE, format='OGG', subtype='VORBIS', compression_level=.5)
        encoded = payload.getvalue()
        y, rate = sf.read(BytesIO(encoded), dtype='float32')
        if rate != RATE or len(y) != len(x) or y.ndim != 1:
            raise ValueError('Codec changed sample count/rate/channels; label timing cannot be assumed')
        if not np.any(x):
            raise ValueError('Codec alignment undefined for silent source')
        lags = correlation_lags(len(y), len(x))
        correlation = correlate(y, x, method='fft')
        lag = int(lags[int(np.argmax(correlation))])
        if lag != 0:
            raise ValueError('Codec decoded alignment is not zero lag; do not silently shift labels')
        metadata.update(codec='OGG/VORBIS', compression_level=.5, soundfile_version=sf.__version__,
                        libsndfile_version=sf.__libsndfile_version__, decoded_global_correlation_lag_samples=lag,
                        encoded_sha256=sha256(encoded).hexdigest(), encoded_bytes=len(encoded), codec_payload=encoded,
                        limitation='Global waveform alignment check does not prove local perceptual boundary identity')
    if not np.isfinite(y).all():
        raise ValueError('Channel transform produced nonfinite samples')
    metadata.update(output_samples=len(y), output_channels=1 if y.ndim == 1 else y.shape[1],
                    samples_above_pcm16_range=int(np.count_nonzero((y < -1)|(y > 32767/32768))))
    return y, labels, metadata
