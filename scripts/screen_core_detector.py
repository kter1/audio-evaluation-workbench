"""Run the frozen core voice detector on a previously acquired small development set."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.core_screen import screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--detector', choices=('fusion', 'wav2vec2', 'rawnetlite', 'pella-current', 'wavlm-aasist'), default='fusion',
                        help='Research comparison only; does not change the app detector')
    parser.add_argument('--stress', action='store_true', help='Fixed original/gain/noise/bandwidth/Vorbis conditions, applied to both classes')
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('threads must be 1..4')
    try:
        raw = args.acquisition.read_bytes()
        acquisition = json.loads(raw)
        if acquisition['spec']['dataset'] != 'interspeech2712/voxenes-2026' or acquisition['spec']['version'] != 1:
            raise ValueError('Unrecognized source acquisition')
        if not 1 <= len(acquisition['files']) <= 50:
            raise ValueError('Screening is bounded to 1..50 clips')
        import torch
        started = time.perf_counter()
        if args.detector == 'fusion':
            from lab.fusion import FusionDetector
            detector = FusionDetector()
        elif args.detector == 'wav2vec2':
            from lab.forensics import Detector, POLICY
            detector = Detector()
            detector.threshold = POLICY['binary_threshold']
        elif args.detector == 'rawnetlite':
            from lab.research_rawnetlite import RawNetLiteDetector
            detector = RawNetLiteDetector()
        elif args.detector == 'pella-current':
            from lab.research_pella_current import PellaCurrentDetector
            detector = PellaCurrentDetector()
        else:
            from lab.research_wavlm_aasist import WavLMAASISTDetector
            detector = WavLMAASISTDetector()
        torch.set_num_threads(args.threads)
        model_load_seconds = time.perf_counter()-started
        result = screen(acquisition, ROOT, detector, stress=args.stress)
        result['acquisition_sha256'] = sha256(raw).hexdigest()
        result['detector_key'] = args.detector
        result['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                                 'threads': torch.get_num_threads(), 'platform': platform.platform()}
        result['runtime_including_model_load_seconds'] = time.perf_counter()-started
        result['model_construction_and_verified_load_seconds'] = model_load_seconds
        if hasattr(detector, 'timings'):
            import resource
            import numpy as np
            timing = detector.timings
            total = sum(t['seconds'] for t in timing)
            result['resources'] = {'score_calls': len(timing), 'parameters': detector.parameters,
                'score_latency_p50_seconds': float(np.quantile([t['seconds'] for t in timing], .5)),
                'score_latency_p95_seconds': float(np.quantile([t['seconds'] for t in timing], .95)),
                'score_seconds_total': total,
                'score_realtime_factor_supplied_audio': total/sum(t['input_seconds'] for t in timing),
                'score_realtime_factor_fixed_model_audio': total/sum(t['model_seconds'] for t in timing),
                'process_peak_resident_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform == 'darwin' else 1024**2),
                'scope': 'Sequential CPU calls including first warm-up; excludes shared decode/transforms/quality checks. Peak RSS covers entire process. Not product latency.'}
        result['code_sha256'] = {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('lab/core_screen.py', 'scripts/screen_core_detector.py', 'lab/fusion.py', 'lab/pella.py', 'lab/antideepfake.py', 'lab/forensics.py', 'lab/vad_stress.py', 'lab/vad_channel_stress.py')}
        if args.detector == 'rawnetlite':
            for name in ('lab/research_rawnetlite.py', 'lab/vendor/rawnetlite_model.py', 'docs/rawnetlite-acquisition-v1.json'):
                result['code_sha256'][name] = sha256((ROOT/name).read_bytes()).hexdigest()
        if args.detector == 'pella-current':
            for name in ('lab/research_pella_current.py', 'docs/pella-current-acquisition-v1.json', 'docs/pella-current-runtime-v1.json'):
                result['code_sha256'][name] = sha256((ROOT/name).read_bytes()).hexdigest()
        if args.detector == 'wavlm-aasist':
            for name in ('lab/research_wavlm_aasist.py', 'docs/wavlm-aasist-acquisition-v1.json'):
                result['code_sha256'][name] = sha256((ROOT/name).read_bytes()).hexdigest()
        text = json.dumps(result, indent=2, allow_nan=False)+'\n'
        directory = ROOT/'data/evaluation/method-runs'
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory/(sha256(text.rstrip('\n').encode()).hexdigest()+'.json')
        if not receipt.exists():
            with receipt.open('x') as stream: stream.write(text)
        elif receipt.read_text() != text:
            raise ValueError('Prior-use receipt content mismatch')
        with args.output.open('x') as stream: stream.write(text)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f'Core screen failed: {exc}\n')
    print('Prior-use receipt:', receipt.relative_to(ROOT))
    for name, counts in result['by_group'].items(): print(name, counts)


if __name__ == '__main__':
    main()
