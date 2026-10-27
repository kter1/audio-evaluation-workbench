"""Fresh-process CPU timing of already-consumed development audio; no downloads."""
import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--comparison', required=True, type=Path, help='Existing comparison that already consumed this manifest')
    parser.add_argument('--method', required=True, choices=('energy', 'webrtc', 'silero'))
    parser.add_argument('--silero-model', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('repeats must lie in 1..10')
    if args.method == 'silero' and args.silero_model is None:
        parser.error('silero requires --silero-model')
    try:
        comparison_raw = args.comparison.read_bytes()
        comparison = json.loads(comparison_raw)
        if (comparison.get('schema_version') != 'method-benchmark-result-1.0'
                or comparison.get('mode') != 'supplied development annotations'
                or comparison.get('manifest_sha256') != sha256(args.manifest.read_bytes()).hexdigest()):
            raise ValueError('Matching prior development comparison required')
        import_start = time.perf_counter()
        from lab.research_benchmark import load_manifest
        from lab.research_analysis import energy_activity, optional_webrtc, RATE
        from lab.runtime_metrics import peak_rss_gib, timing_summary
        import_seconds = time.perf_counter()-import_start
        data_start = time.perf_counter()
        cases = load_manifest(args.manifest, ROOT)
        previous = {r['id']: r for r in comparison['results']}
        if len(previous) != len(cases) or set(previous) != {c['id'] for c in cases}:
            raise ValueError('Prior comparison must cover every runtime case exactly once')
        for case in cases:
            if sha256(case['samples'].astype('<f4').tobytes()).hexdigest() != previous[case['id']]['analysis_samples_sha256']:
                raise ValueError('Runtime audio differs from prior comparison')
        data_seconds = time.perf_counter()-data_start
        setup_start = time.perf_counter()
        model_load_seconds = None
        model = None
        if args.method == 'silero':
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from lab.research_vad import SileroSession
            load_start = time.perf_counter()
            model = SileroSession(args.silero_model)
            model_load_seconds = time.perf_counter()-load_start
            predict = model.predict
        elif args.method == 'webrtc':
            importlib.import_module('webrtcvad')
            predict = optional_webrtc
        else:
            predict = energy_activity
        setup_seconds = time.perf_counter()-setup_start
        warmup_start = time.perf_counter()
        warmup = predict(cases[0]['samples'])
        if isinstance(warmup, dict) and warmup.get('status') != 'available':
            raise ValueError('Selected method is unavailable')
        warmup_seconds = time.perf_counter()-warmup_start
        rows = []
        for repeat in range(args.repeats):
            for case in cases:
                begin = time.perf_counter()
                prediction = predict(case['samples'])
                elapsed = time.perf_counter()-begin
                intervals = prediction['intervals'] if isinstance(prediction, dict) else prediction
                prior_key = {'energy': 'energy_activity_diagnostic', 'webrtc': 'webrtc', 'silero': 'silero_raw'}[args.method]
                # Energy comparisons store only metrics; optional adapters store intervals.
                prior = previous[case['id']]['speech_methods'][prior_key]
                if args.method != 'energy' and intervals != prior['intervals']:
                    raise ValueError('Runtime predictions differ from prior comparison')
                rows.append({'id': case['id'], 'repeat': repeat, 'audio_seconds': len(case['samples'])/RATE,
                             'wall_seconds': elapsed})
        hardware = {}
        if platform.system() == 'Darwin':
            for key in ('hw.model', 'machdep.cpu.brand_string'):
                query = subprocess.run(['sysctl', '-n', key], text=True, capture_output=True, check=False)
                hardware[key] = query.stdout.strip() if query.returncode == 0 else {'unavailable': query.stderr.strip()}
        elif platform.system() == 'Linux':
            hardware['cpuinfo'] = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
        memory = peak_rss_gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, platform.system())
        summary = timing_summary(rows)
        result = {'schema_version': 'vad-runtime-1.0', 'method': args.method,
            'manifest_sha256': comparison['manifest_sha256'], 'comparison_sha256': sha256(comparison_raw).hexdigest(),
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                'hardware': hardware, 'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
                'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                'device': 'cpu', 'torch_threads': torch.get_num_threads() if model else None,
                'torch_interop_threads': torch.get_num_interop_threads() if model else None},
            'case_count': len(cases), 'repeats': args.repeats,
            'cold_module_import_seconds': import_seconds, 'decode_and_verify_seconds': data_seconds,
            'cold_adapter_setup_seconds': setup_seconds, 'model_load_and_verification_seconds': model_load_seconds,
            'first_recording_warmup_seconds': warmup_seconds, 'warm_summary': summary,
            'peak_process_rss_gib': memory, 'worker_wall_seconds': time.perf_counter()-started,
            'resource_checks_only': {'warm_realtime_factor_le_1': summary['realtime_factor'] <= 1,
                                     'peak_process_rss_le_8_gib': memory <= 8},
            'model_sha256': model.model_sha256 if model else None, 'rows': rows,
            'code_sha256': {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('scripts/benchmark_vad_runtime.py', 'lab/runtime_metrics.py', 'lab/research_vad.py', 'lab/research_analysis.py', 'lab/research_benchmark.py')},
            'scope': 'Local fresh-process CPU experiment on previously consumed development clips; no accuracy or promotion claim. '
                     'One first-clip warmup excluded. Timed calls include per-recording state reset, validation and output conversion. '
                     'Peak RSS includes interpreter, imports and all decoded clips; not model-only memory. '
                     'Worker wall excludes interpreter startup and output writing. Repeats are not independent recordings. '
                     'Run each method in a separate invocation; shared model is sequential and not thread safe.'}
        if args.method == 'webrtc':
            result['package_version'] = warmup['package_version']
        elif model:
            result['package_version'] = torch.__version__
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f'Runtime benchmark failed: {exc}\n')
    print(f"{args.method}: warm RTF {summary['realtime_factor']:.5f}; peak RSS {memory:.3f} GiB")


if __name__ == '__main__':
    main()
