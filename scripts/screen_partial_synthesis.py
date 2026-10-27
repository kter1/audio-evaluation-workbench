"""Small fixed, constructed partial-synthesis challenge on spent v4 sources."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.baseline_audit import checked_path
from lab.corpus_inventory import connected_groups
from lab.forensics import (POLICY, RATE, abstention_band, anchored_spans, decision,
                           diagnostics, mono16, read_audio)
from lab.store import digest
from scripts.screen_candidate_regression import write_receipt

SEED = '20261023:partial:'
CONTEXT_SAMPLES = 12*RATE
INSERT_SECONDS = (1, 2, 4)


def selection(document):
    """Fix four distinct contexts and genuine controls using metadata alone."""
    if (document.get('version') != '4.0' or not document.get('cases')
            or digest({k: v for k, v in document.items() if k != 'sha256'}) != document.get('sha256')):
        raise ValueError('Invalid frozen v4 manifest or self-digest')
    cases = document['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate v4 case IDs')
    humans = [c for c in cases if c['label'] == 'human']
    candidates = []
    for fake in (c for c in cases if c['label'] == 'synthetic' and c['duration'] >= 4):
        for human in humans:
            if human['source_utt'] != fake['source_utt'] or human['speaker'] != fake['speaker'] or human['duration'] < 12:
                continue
            controls = [c for c in humans if c['speaker'] == human['speaker']
                        and c['source_utt'] != human['source_utt'] and c['duration'] >= 4]
            if controls:
                control = min(controls, key=lambda c: sha256((SEED+'control:'+c['id']).encode()).hexdigest())
                pair_id = human['id']+':'+fake['id']
                candidates.append((sha256((SEED+pair_id).encode()).hexdigest(), pair_id, human, fake, control))
    selected, used_contexts = [], set()
    for _, pair_id, human, fake, control in sorted(candidates):
        if human['id'] in used_contexts:
            continue
        selected.append((pair_id, human, fake, control))
        used_contexts.add(human['id'])
        if len(selected) == 4:
            return selected
    raise ValueError('Need four distinct >=12-second matched genuine contexts and same-speaker genuine controls')


def center(samples, count):
    if len(samples) < count:
        raise ValueError('Source shorter than required context or insert; no repetition or padding allowed')
    start = (len(samples)-count)//2
    return samples[start:start+count].copy(), start


def prepare_challenge(document, root):
    chosen = selection(document)  # All identities are selected before decoding or inference.
    root = Path(root).resolve()
    sources = {c['id']: c for _, *triple in chosen for c in triple}
    decoded, lineage = {}, {}
    # Authenticate ALL selected files before constructing or scoring any case.
    for source_id, case in sources.items():
        path = checked_path(root, case['path'])
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != case['sha256']:
            raise ValueError('Partial-challenge source hash mismatch: '+source_id)
        audio, rate, header = read_audio(raw)
        decoded[source_id] = mono16(audio, rate)
        lineage[source_id] = {'id': source_id, 'path': str(path.relative_to(root)),
            'sha256': case['sha256'], 'source_utt': case['source_utt'], 'speaker': case['speaker'],
            'ref_utt': case.get('ref_utt'), 'source_header': header}
    inputs, identities, pairs = [], [], []
    for pair_id, human, fake, control in chosen:
        context, context_start = center(decoded[human['id']], CONTEXT_SAMPLES)
        tokens = {'librispeech:speaker:'+str(human['speaker'])}
        for case in (human, fake, control):
            tokens.update(('audio:'+case['sha256'], 'librispeech:utterance:'+case['source_utt']))
            if case.get('ref_utt'):
                tokens.add('librispeech:utterance:'+case['ref_utt'])
        pair = {'pair_id': pair_id, 'genuine_context': lineage[human['id']],
            'synthetic_insert_source': lineage[fake['id']], 'genuine_insert_source': lineage[control['id']],
            'generator': fake['generator'], 'context_start_sample_16k': context_start}
        pairs.append(pair)
        variants = [('original', 'pristine_genuine', context.copy(), None)]
        for seconds in INSERT_SECONDS:
            count = seconds*RATE
            destination_start = (CONTEXT_SAMPLES-count)//2
            for name, role, source_case in (('partial', 'partial_synthetic', fake), ('genuine_splice', 'genuine_splice', control)):
                insert, insert_start = center(decoded[source_case['id']], count)
                samples = context.copy()
                samples[destination_start:destination_start+count] = insert
                variants.append((f'{name}_{seconds}s', role, samples,
                    {'start_sample_16k': destination_start, 'end_sample_16k': destination_start+count,
                     'source_id': source_case['id'], 'source_start_sample_16k': insert_start,
                     'seconds': seconds, 'boundary_policy': 'Hard splice at both boundaries, also applied to genuine controls; no gain normalization or crossfade'}))
        for condition, role, samples, splice in variants:
            record_id = 'v4-partial:'+pair_id+'::'+condition
            pcm_hash = sha256(samples.astype('<f4').tobytes()).hexdigest()
            identities.append({'record_id': record_id,
                'identity_tokens': sorted(tokens | {'analysis-pcm:'+pcm_hash})})
            inputs.append((samples, {'id': record_id, 'pair_id': pair_id, 'condition': condition,
                'challenge_role': role, 'label': 'synthetic' if role == 'partial_synthetic' else 'human',
                'generator': fake['generator'] if role == 'partial_synthetic' else 'bonafide',
                'analysis_samples_sha256': pcm_hash, 'duration_seconds': 12., 'splice': splice}))
    groups = connected_groups(identities)
    indexed_groups = {record: sha256('\n'.join(group).encode()).hexdigest()[:16]
                      for group in groups for record in group}
    for _, metadata in inputs:
        metadata['group'] = indexed_groups[metadata['id']]
    return inputs, identities, pairs


def summarize_outcomes(rows):
    partial = [r for r in rows if r['challenge_role'] == 'partial_synthetic']
    genuine = [r for r in rows if r['challenge_role'] != 'partial_synthetic']
    return {'cases': len(rows), 'partial_synthetic': len(partial), 'genuine_controls': len(genuine),
        'false_human_indications_on_partial': sum(r['assessment']['assessment'] == 'Human speech indicated' for r in partial),
        'false_synthetic_indications_on_genuine_controls': sum(r['assessment']['assessment'] == 'Synthetic speech indicated' for r in genuine),
        'synthetic_indications_on_partial': sum(r['assessment']['assessment'] == 'Synthetic speech indicated' for r in partial),
        'inconclusive_partial': sum(r['assessment']['assessment'] == 'Inconclusive' for r in partial),
        'human_indications_on_genuine_controls': sum(r['assessment']['assessment'] == 'Human speech indicated' for r in genuine),
        'inconclusive_genuine_controls': sum(r['assessment']['assessment'] == 'Inconclusive' for r in genuine)}


def screen(document, root, detector, detector_key='research_candidate'):
    inputs, identities, pairs = prepare_challenge(document, root)
    threshold = float(detector.threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Finite threshold in (0,1) required')
    window_samples = getattr(detector, 'window_samples', 4*RATE)
    rows = []
    for samples, metadata in inputs:
        _, metrics, warnings = diagnostics(samples[:, None], RATE)
        spans = detector.spans(samples) if hasattr(detector, 'spans') else anchored_spans(len(samples), window_samples)
        if spans != anchored_spans(len(samples), window_samples):
            raise ValueError('Complete anchored native-window coverage required')
        windows = []
        for start, end in spans:
            windows.append({'start': start/RATE, 'end': end/RATE, 'aggregation_weight': 1.,
                'analysis_samples_sha256': sha256(samples[start:end].astype('<f4').tobytes()).hexdigest(),
                **detector.score(samples[start:end])})
        assessment = decision(windows, metrics, len(samples)/RATE, threshold,
                              abstention_band(threshold, getattr(detector, 'band_logit', None)))
        assessment['aggregation'] = 'Arithmetic mean of complete native-size anchored window scores'
        rows.append({**metadata, 'synthetic_score': assessment['synthetic_score'],
            'assessment': assessment, 'windows': windows, 'diagnostics': metrics, 'warnings': warnings})
    return {'schema_version': 'core-detector-screen-1.0', 'mode': 'supplied development authenticity labels',
        'detector_key': detector_key, 'model': detector.manifest, 'threshold': threshold, 'policy': POLICY,
        'manifest_self_digest': document['sha256'], 'case_count': len(rows), 'original_context_count': 4,
        'source_groups': len({r['group'] for r in rows}), 'input_identities': identities,
        'selection': "First four distinct genuine contexts ordered by SHA256('20261023:partial:'+humanID+':'+syntheticID); >=12s context, >=4s synthetic source, same-speaker/source_utt pair; genuine alternate source chosen by SHA256('20261023:partial:control:'+ID)",
        'construction': {'context_samples': CONTEXT_SAMPLES, 'sample_rate': RATE, 'insert_seconds': list(INSERT_SECONDS),
            'context_and_excerpt_policy': 'Center by sample count after existing mono16 conversion; no repetition, padding, gain normalization or crossfade',
            'position': 'Centered in 12-second context, identical boundaries for partial and genuine splice control',
            'candidate_window_samples': window_samples, 'candidate_aggregation': 'Equal mean of anchored native-size windows'},
        'pairs': pairs, 'results': rows, 'outcomes': summarize_outcomes(rows),
        'by_condition': {c: summarize_outcomes([r for r in rows if r['condition'] == c]) for c in sorted({r['condition'] for r in rows})},
        'scope': 'Small constructed challenge on spent English v4 sources. Known synthetic insert presence is the reference, not a claim about natural mixed-audio detection or edit localization. '
                 'Genuine alternate-utterance splices control for both hard boundaries and inherited level differences. '
                 'The four contexts and seven variants are correlated, with source/speaker/cloning-reference lineage preserved. '
                 'Current core quality and abstention checks only; no product perturbation-stability check, fitting, threshold tuning or model promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector', choices=('fusion', 'wavlm-aasist'), default='fusion')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('threads must be 1..4')
    try:
        if args.output.exists():
            raise ValueError('Output already exists; choose a new filename')
        manifest_raw = (ROOT/'data/evaluation/v4-manifest.json').read_bytes()
        document = json.loads(manifest_raw)
        if len(document['cases']) != 320:
            raise ValueError('Require complete frozen v4 source manifest')
        prepare_challenge(document, ROOT)  # Source/selection checks precede model loading.
        import torch
        torch.set_num_threads(args.threads)
        started = time.perf_counter()
        if args.detector == 'fusion':
            from lab.fusion import FusionDetector
            detector = FusionDetector()
            adapter_path = 'lab/fusion.py'
        else:
            from lab.research_wavlm_aasist import WavLMAASISTDetector
            detector = WavLMAASISTDetector()
            adapter_path = 'lab/research_wavlm_aasist.py'
        torch.set_num_threads(args.threads)
        load_seconds = time.perf_counter()-started
        result = screen(document, ROOT, detector, args.detector)
        result.update({'manifest_file_sha256': sha256(manifest_raw).hexdigest(),
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                'threads': torch.get_num_threads(), 'platform': platform.platform()},
            'runtime_including_model_load_seconds': time.perf_counter()-started,
            'model_construction_and_verified_load_seconds': load_seconds,
            'code_sha256': {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('scripts/screen_partial_synthesis.py', 'scripts/screen_candidate_regression.py',
                 adapter_path, 'lab/forensics.py', 'lab/corpus_inventory.py')}})
        receipt = write_receipt(result, args.output, ROOT/'data/evaluation/method-runs')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f'Partial synthesis challenge failed: {exc}\n')
    print('Prior-use receipt:', receipt.relative_to(ROOT))
    print(json.dumps(result['outcomes'], indent=2))


if __name__ == '__main__':
    main()
