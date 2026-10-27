"""Check fixed CPU/MPS numerical parity before accelerated spent-v4 regression."""
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

from lab.core_screen import prepare_case
from lab.research_wavlm_aasist import CHECKPOINT, CHECKPOINT_SHA256, WavLMAASISTDetector
from lab.research_wavlm_mps import WavLMMPSDetector
from scripts.screen_candidate_regression import (
    prepare_manifest, regression, validate_baseline, write_receipt,
)

PROBABILITY_TOLERANCE = 1e-4
LOGIT_TOLERANCE = 1e-3
STORED_CPU_TOLERANCE = 1e-6
CONDITIONS = ('original', 'gain_minus12db', 'added_noise20db', 'bandwidth8k', 'vorbis_roundtrip')
CODE_PATHS = (
    'lab/research_wavlm_aasist.py', 'lab/research_wavlm_mps.py',
    'scripts/screen_wavlm_mps.py', 'scripts/screen_candidate_regression.py',
    'lab/core_screen.py', 'lab/forensics.py', 'lab/corpus_inventory.py',
    'lab/vad_stress.py', 'lab/vad_channel_stress.py',
)


def hashes():
    return {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in CODE_PATHS}


def selected_rows(screen):
    """Fixed ten strata: both labels in every condition, all five AI families."""
    if (screen.get('detector_key') != 'wavlm-aasist'
            or screen.get('case_count') != 225 or len(screen.get('results', [])) != 225
            or screen.get('threshold') != .5
            or screen.get('model', {}).get('files', {}).get(CHECKPOINT, {}).get('sha256') != CHECKPOINT_SHA256):
        raise ValueError('Require the complete frozen 225-case CPU candidate screen')
    families = sorted({r['group'] for r in screen['results'] if r['label'] == 'synthetic'})
    if families != ['chroma', 'glm_tts', 'qwen3_tts', 'vibevoice', 'voxcpm']:
        raise ValueError('Require the five declared modern-generator families')
    selected = []
    for index, condition in enumerate(CONDITIONS):
        for label, group in (('human', 'human'), ('synthetic', families[index])):
            eligible = [r for r in screen['results']
                        if r['label'] == label and r['group'] == group and r['condition'] == condition]
            if not eligible:
                raise ValueError('Missing fixed parity stratum')
            selected.append(min(eligible, key=lambda row: row['id']))
    if len({r['id'] for r in selected}) != 10:
        raise ValueError('Duplicate parity selections')
    return selected


def reconstructed(row, acquisition):
    item = next((item for item in acquisition['files']
                 if row['id'] == 'voxenes-2026:'+item['name']+'::'+row['condition']), None)
    if item is None:
        raise ValueError('Parity source is absent from pinned acquisition')
    samples, _, _, provenance = prepare_case(item, ROOT)
    if provenance['analysis_samples_sha256'] != row['analysis_samples_sha256']:
        raise ValueError('Parity original analysis PCM hash mismatch')
    if row['condition'] != 'original':
        from lab.vad_stress import transform
        seed = int(sha256((item['sha256']+':core-screen-1.0:'+row['condition']).encode()).hexdigest()[:16], 16)
        samples, _, _ = transform(samples, None, row['condition'], seed)
    if sha256(samples.astype('<f4').tobytes()).hexdigest() != row['transformed_samples_sha256']:
        raise ValueError('Parity transformed PCM hash mismatch')
    return samples


def comparison(cpu, gpu, stored):
    probability_error = abs(cpu['synthetic_score']-gpu['synthetic_score'])
    logit_error = abs(cpu['native_real_logit']-gpu['native_real_logit'])
    stored_error = abs(cpu['synthetic_score']-stored['synthetic_score'])
    input_matches = cpu['model_input_sha256'] == gpu['model_input_sha256'] == stored['model_input_sha256']
    binary_matches = ((cpu['synthetic_score'] >= .5) == (gpu['synthetic_score'] >= .5)
                      == (stored['synthetic_score'] >= .5))
    return {'probability_abs_error': probability_error, 'native_logit_abs_error': logit_error,
            'stored_cpu_probability_abs_error': stored_error,
            'model_input_hash_matches': input_matches, 'binary_matches': binary_matches,
            'passed': bool(input_matches and binary_matches
                           and probability_error <= PROBABILITY_TOLERANCE
                           and logit_error <= LOGIT_TOLERANCE
                           and stored_error <= STORED_CPU_TOLERANCE)}


def verify(model, screen, acquisition, torch):
    selected = selected_rows(screen)
    inputs = [reconstructed(row, acquisition) for row in selected]
    # Stored screen retains probabilities, not native logits. Recompute only the
    # ten fixed CPU references, checking every one against the frozen CPU score.
    model.model.to('cpu')
    cpu = [WavLMAASISTDetector.score(model, samples) for samples in inputs]
    model.model.to('mps')
    torch.mps.synchronize()
    model.timings = []
    gpu = [model.score(samples) for samples in inputs]
    rows = [{'id': original['id'], 'label': original['label'], 'group': original['group'],
             'condition': original['condition'], 'cpu': reference, 'mps': accelerated,
             **comparison(reference, accelerated, original)}
            for original, reference, accelerated in zip(selected, cpu, gpu)]
    return {'schema_version': 'wavlm-device-parity-1.0', 'passed': all(row['passed'] for row in rows),
            'checkpoint_sha256': CHECKPOINT_SHA256, 'model': model.manifest,
            'selection': 'Ten fixed lexicographically first cases: human in every declared condition plus each of five AI families in one condition; selection independent of scores',
            'case_count': 10, 'results': rows,
            'tolerances': {'probability_abs': PROBABILITY_TOLERANCE,
                           'native_real_logit_abs': LOGIT_TOLERANCE,
                           'stored_cpu_probability_abs': STORED_CPU_TOLERANCE,
                           'identical_model_input_hashes': True, 'identical_binary_decisions': True},
            'scope': 'Numerical device parity on ten previously scored development examples; not an accuracy estimate or independent validation. CPU references are recomputed because the stored screen omits native logits.'}


def validate_parity(report, screen_raw):
    if (report.get('schema_version') != 'wavlm-device-parity-1.0'
            or report.get('passed') is not True or report.get('case_count') != 10
            or report.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or report.get('code_sha256') != hashes()
            or report.get('cpu_screen_file_sha256') != sha256(screen_raw).hexdigest()
            or report.get('tolerances') != {'probability_abs': PROBABILITY_TOLERANCE,
                    'native_real_logit_abs': LOGIT_TOLERANCE,
                    'stored_cpu_probability_abs': STORED_CPU_TOLERANCE,
                    'identical_model_input_hashes': True, 'identical_binary_decisions': True}):
        raise ValueError('MPS regression requires a passing parity report for this exact code, checkpoint and CPU screen')
    selected = selected_rows(json.loads(screen_raw))
    rows = report.get('results', [])
    if ([row.get('id') for row in rows] != [row['id'] for row in selected]
            or not all(row.get('passed') is True and comparison(row['cpu'], row['mps'], stored)['passed']
                       for row, stored in zip(rows, selected))):
        raise ValueError('Parity report rows fail fixed selection or numerical gates')


def resources(model, torch):
    import resource
    timings = model.timings
    seconds = [t['seconds'] for t in timings]
    return {'calls': len(timings), 'seconds_total': sum(seconds),
            'latency_p50_seconds': float(np.quantile(seconds, .5)),
            'latency_p95_seconds': float(np.quantile(seconds, .95)),
            'realtime_factor_window_inputs': sum(seconds)/sum(t['input_seconds'] for t in timings),
            'process_peak_resident_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform == 'darwin' else 1024**2),
            'mps_current_allocated_gib': torch.mps.current_allocated_memory()/1024**3,
            'mps_driver_allocated_gib': torch.mps.driver_allocated_memory()/1024**3,
            'scope': 'Sequential synchronized MPS calls including first call. RSS includes CPU/model load; driver allocation is final resident allocation, not peak GPU memory.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'regression'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parity-report', type=Path)
    parser.add_argument('--cpu-screen', type=Path, default=ROOT/'docs/core-wavlm-aasist-screen-v1.json')
    parser.add_argument('--acquisition', type=Path, default=ROOT/'docs/core-modern-acquisition-v1.json')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Output already exists; choose a new filename')
        screen_raw, acquisition_raw = args.cpu_screen.read_bytes(), args.acquisition.read_bytes()
        screen, acquisition = json.loads(screen_raw), json.loads(acquisition_raw)
        if sha256(acquisition_raw).hexdigest() != screen['acquisition_sha256']:
            raise ValueError('Acquisition differs from the frozen CPU screen')
        if args.mode == 'regression':
            if args.parity_report is None:
                raise ValueError('A passing --parity-report is required')
            parity_raw = args.parity_report.read_bytes()
            validate_parity(json.loads(parity_raw), screen_raw)
            manifest_raw = (ROOT/'data/evaluation/v4-manifest.json').read_bytes()
            baseline_raw = (ROOT/'docs/detector-evaluation.json').read_bytes()
            manifest, baseline = json.loads(manifest_raw), json.loads(baseline_raw)
            if len(manifest['cases']) != 320:
                raise ValueError('MPS regression requires complete 320-original manifest')
            prepare_manifest(manifest, ROOT)
            validate_baseline(baseline, manifest)
        import torch
        torch.set_num_threads(1)
        started = time.perf_counter()
        model = WavLMMPSDetector()
        load_seconds = time.perf_counter()-started
        if args.mode == 'verify':
            result = verify(model, screen, acquisition, torch)
        else:
            result = regression(manifest, ROOT, model, baseline, None,
                                lambda n, total: print(f'MPS regression {n}/{total}', flush=True)
                                if n % 40 == 0 or n == total else None,
                                detector_key='wavlm-aasist-mps')
            result.update({'manifest_file_sha256': sha256(manifest_raw).hexdigest(),
                           'historical_baseline_file_sha256': sha256(baseline_raw).hexdigest(),
                           'parity_report_sha256': sha256(parity_raw).hexdigest()})
        result.update({'code_sha256': hashes(),
                       'cpu_screen_file_sha256': sha256(screen_raw).hexdigest(),
                       'acquisition_sha256': sha256(acquisition_raw).hexdigest(),
                       'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                       'threads': torch.get_num_threads(), 'platform': platform.platform(),
                                       'device': 'mps', 'mps_built': torch.backends.mps.is_built()},
                       'model_construction_and_verified_load_seconds': load_seconds,
                       'runtime_including_model_load_seconds': time.perf_counter()-started,
                       'score_resources': resources(model, torch)})
        if args.mode == 'regression':
            receipt = write_receipt(result, args.output, ROOT/'data/evaluation/method-runs')
            print('Prior-use receipt:', receipt.relative_to(ROOT))
        else:
            with args.output.open('x') as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print('MPS parity:', result['passed'], f"{sum(row['passed'] for row in result['results'])}/10")
            if not result['passed']:
                parser.exit(2, 'MPS parity failed; regression is disabled\n')
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'MPS research run failed: {exc}\n')


if __name__ == '__main__':
    main()
