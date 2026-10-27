"""Bounded deterministic development perturbations; inherited labels are not new truth."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from lab.baseline_audit import checked_path
from lab.research_metrics import validate_intervals

RATE = 16000
CONDITIONS = ('original', 'gain_minus12db', 'gain_minus24db', 'added_noise20db', 'added_noise10db', 'prefix137ms')


def transform(samples, intervals, condition, seed):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or len(x) < RATE//2 or not np.isfinite(x).all():
        raise ValueError('Finite mono 16 kHz recording of at least 0.5 s required')
    if intervals is not None:
        validate_intervals(intervals, len(x)/RATE, nonoverlap=True)
    if condition not in CONDITIONS:
        from lab.vad_channel_stress import transform_channel
        return transform_channel(x, intervals, condition, seed)
    labels = deepcopy(intervals)
    metadata = {'condition': condition, 'seed': seed, 'input_samples': len(x),
                'label_policy': 'Inherited latent speech timing; transformed audibility not independently verified'}
    if condition.startswith('gain_'):
        gain = -12 if condition == 'gain_minus12db' else -24
        y = x*10**(gain/20)
        metadata['gain_db'] = gain
    elif condition.startswith('added_noise'):
        ratio = 20 if condition == 'added_noise20db' else 10
        rms = float(np.sqrt(np.mean(x.astype(float)**2)))
        if rms == 0:
            raise ValueError('Mixture-relative noise scaling undefined for zero-RMS source')
        noise = np.random.Generator(np.random.PCG64(seed)).normal(size=len(x))
        noise *= rms*10**(-ratio/20)/np.sqrt(np.mean(noise**2))
        y = (x.astype(float)+noise).astype(np.float32)
        metadata.update(mixture_to_added_noise_db=ratio, rng='PCG64',
                        added_noise_rms=float(np.sqrt(np.mean(noise**2))),
                        ratio_scope='Whole original mixture RMS / constructed noise RMS; not clean-speech SNR')
    elif condition == 'prefix137ms':
        prefix = 2192
        y = np.r_[np.zeros(prefix, np.float32), x]
        if labels is not None:
            labels = [{**i, 'start': i['start']+prefix/RATE, 'end': min(len(y)/RATE, i['end']+prefix/RATE)} for i in labels]
        metadata.update(prefix_samples=prefix, prefix_label='constructed exact digital silence')
    else:
        y = x.copy()
    metadata.update(output_samples=len(y), samples_above_pcm16_range=int(np.count_nonzero((y < -1)|(y > 32767/32768))))
    return y, labels, metadata


def build_stress(manifest_path, root, destination, extended=False):
    if extended:
        from lab.vad_channel_stress import CHANNEL_CONDITIONS
        conditions, version = CHANNEL_CONDITIONS, 'vad-channel-stress-1.0'
    else:
        conditions, version = CONDITIONS, 'vad-stress-1.0'
    root, destination = Path(root).resolve(), Path(destination).absolute()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError('Stress output must be inside workspace') from exc
    if destination.exists():
        raise ValueError('Destination must not exist')
    raw = Path(manifest_path).read_bytes()
    doc = json.loads(raw)
    rows = doc.get('cases', [])
    if doc.get('schema_version') != 'method-benchmark-1.0' or doc.get('partition') != 'development':
        raise ValueError('Only development manifests are allowed')
    if not rows or len(rows) > 50 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Require 1..50 uniquely identified source cases')
    total = 0.
    for row in rows:
        if row.get('partition', 'development') != 'development':
            raise ValueError('All cases must be development')
        info = sf.info(checked_path(root, row['path']))
        if info.samplerate != RATE or info.channels != 1 or not .5 <= info.duration <= 60:
            raise ValueError('Stress inputs must be mono 16 kHz and 0.5..60 seconds')
        total += info.duration
    if total > 600:
        raise ValueError('Source audio exceeds 600-second budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.vad-stress-', dir=destination.parent))
    try:
        cases = []
        for index, row in enumerate(rows):
            source = checked_path(root, row['path'])
            raw_audio = source.read_bytes()
            if sha256(raw_audio).hexdigest() != row['sha256']:
                raise ValueError('Source audio hash mismatch')
            x, _ = sf.read(source, dtype='float32')
            for condition in conditions:
                seed = int(sha256((row['sha256']+':'+version+':'+condition).encode()).hexdigest()[:16], 16)
                y, labels, metadata = transform(x, row.get('speech_intervals'), condition, seed)
                payload = metadata.pop('codec_payload', None)
                if payload is not None:
                    encoded_name = f'{index:03d}-{condition}.ogg'
                    (stage/encoded_name).write_bytes(payload)
                    metadata['encoded_path'] = str((destination/encoded_name).relative_to(root))
                case = deepcopy(row)
                case.update(id=row['id']+'::'+condition, condition=condition, parent_id=row['id'],
                            parent_sha256=row['sha256'], speech_intervals=labels,
                            source_references=sorted(set(row.get('source_references', [])+['audio:'+row['sha256']])),
                            transformation=metadata, level_events=None, noise_reference_dbfs=None)
                case['annotation_provenance'] = row['annotation_provenance']+'; '+version+' inherited timing, not independently relabeled'
                if 'annotation_sha256' in case:
                    case['publisher_annotation_sha256'] = case.pop('annotation_sha256')
                if condition != 'original':
                    name = f'{index:03d}-{condition}.wav'
                    sf.write(stage/name, y, RATE, subtype='FLOAT')
                    case.update(path=str((destination/name).relative_to(root)), sha256=sha256((stage/name).read_bytes()).hexdigest())
                cases.append(case)
        result = {'schema_version': 'method-benchmark-1.0', 'partition': 'development', 'cases': cases,
            'stress_protocol': {'version': version, 'conditions': list(conditions),
                'source_manifest_sha256': sha256(raw).hexdigest(), 'source_case_count': len(rows),
                'numpy_version': np.__version__, 'storage': 'IEEE float32 WAV, no amplitude normalization or clipping',
                'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                'extension_code_sha256': sha256(Path(__file__).with_name('vad_channel_stress.py').read_bytes()).hexdigest() if extended else None,
                'scope': 'Constructed development robustness; original labels inherited, noise is mixture-relative, '
                         'derivatives share original source/speaker groups; no new independent data or accuracy confirmation.'},
            'source_limitations': doc.get('limitations', [])}
        (stage/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        if sum(p.stat().st_size for p in stage.iterdir()) > 256*1024**2:
            raise ValueError('Generated output exceeds 256 MiB budget')
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def summarize_stress(result, manifest):
    from lab.research_vad import summarize_speech
    rows = manifest['cases']
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r['id'] for r in result['results']}:
        raise ValueError('Stress summary requires exact case coverage')
    protocol = manifest['stress_protocol']
    if protocol['version'] == 'vad-channel-stress-1.0':
        from lab.vad_channel_stress import CHANNEL_CONDITIONS
        conditions = CHANNEL_CONDITIONS
    elif protocol['version'] == 'vad-stress-1.0':
        conditions = CONDITIONS
    else:
        raise ValueError('Unknown stress protocol version')
    expected = set(protocol['conditions'])
    if expected != set(conditions) or len(protocol['conditions']) != len(conditions):
        raise ValueError('Unexpected stress protocol conditions')
    parents = {}
    for row in rows:
        entry = parents.setdefault(row['parent_id'], set())
        if row['condition'] in entry:
            raise ValueError('Duplicate parent/condition pair')
        entry.add(row['condition'])
    if any(c != expected for c in parents.values()):
        raise ValueError('Every source must have every stress condition')
    summaries = {}
    for condition in conditions:
        subset = [r for r in result['results'] if by_id[r['id']]['condition'] == condition]
        summaries[condition] = summarize_speech({'results': subset})
    return {'protocol': manifest['stress_protocol'], 'conditions': summaries,
            'scope': 'Per-condition descriptive counts; repeated derivatives are paired, not independent. '
                     'No pooling across conditions as independent support; inherited labels do not prove audibility.'}
