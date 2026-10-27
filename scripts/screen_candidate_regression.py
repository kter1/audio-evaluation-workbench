"""Fixed isolated-candidate regression on the spent v4 corpus; no fitting or promotion."""
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
from lab.core_screen import summarize
from lab.corpus_inventory import connected_groups
from lab.forensics import (POLICY, RATE, abstention_band, anchored_spans, decision,
                           diagnostics, read_audio)
from lab.store import digest

CHANNELS = ('clean', 'mp3', 'phone', 'noise20')


def prepare_manifest(document, root, limit=None):
    """Verify the complete manifest and every selected source before scoring."""
    cases = document.get('cases', [])
    if (document.get('version') != '4.0' or document.get('channels') != list(CHANNELS)
            or not cases or len(cases) > 320
            or digest({k: v for k, v in document.items() if k != 'sha256'}) != document.get('sha256')):
        raise ValueError('Invalid frozen v4 manifest or self-digest')
    if limit is not None and (isinstance(limit, bool) or not 1 <= limit <= len(cases)):
        raise ValueError('Debug limit must be 1..manifest original count')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate v4 case IDs')
    selected = cases if limit is None else cases[:limit]
    inputs, identities = [], []
    for case in selected:
        if (case['label'] not in ('human', 'synthetic') or not case.get('speaker')
                or not case.get('source_utt') or not case.get('generator')):
            raise ValueError('V4 labels, speaker, source utterance and generator are required')
        tokens = {'audio:'+case['sha256'], 'librispeech:speaker:'+str(case['speaker']),
                  'librispeech:utterance:'+case['source_utt']}
        if case.get('ref_utt'):
            tokens.add('librispeech:utterance:'+case['ref_utt'])
        variants = {'clean': case, **case['channels']}
        if set(variants) != set(CHANNELS):
            raise ValueError('Require exactly the four frozen v4 conditions')
        for channel in CHANNELS:
            variant = variants[channel]
            path = checked_path(Path(root).resolve(), variant['path'])
            if sha256(path.read_bytes()).hexdigest() != variant['sha256']:
                raise ValueError(f'V4 audio hash mismatch: {case["id"]}/{channel}')
            tokens.add('audio:'+variant['sha256'])
            inputs.append((case, channel, path, variant['sha256']))
        identities.append({'record_id': 'v4-regression:'+case['id'],
                           'identity_tokens': sorted(tokens)})
    groups = connected_groups(identities)
    group_ids = {record: sha256('\n'.join(group).encode()).hexdigest()[:16]
                 for group in groups for record in group}
    return inputs, identities, group_ids


def totals(rows, threshold):
    grouped = summarize(rows, threshold)
    first = next(iter(grouped.values()))
    return {key: sum(g[key] for g in grouped.values()) for key in first}


def validate_baseline(baseline, document):
    """Match historical raw scores only; do not replay its old assessment policy."""
    if (baseline.get('version') != '4.0'
            or baseline.get('manifest_sha256') != document['sha256']):
        raise ValueError('Historical baseline does not match the frozen v4 manifest')
    rows = baseline['test']['rows']
    indexed = {(r['id'], r['channel']): r for r in rows}
    expected = {(c['id'], ch) for c in document['cases'] for ch in CHANNELS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Historical baseline requires exact complete v4 coverage')
    threshold = baseline.get('threshold')
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Invalid historical threshold')
    labels = {c['id']: c['label'] for c in document['cases']}
    for row in rows:
        if row['label'] not in ('human', 'synthetic') or not np.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Invalid historical raw score')
        if row['label'] != labels[row['id']]:
            raise ValueError('Historical baseline label differs from manifest')
    return indexed


def regression(document, root, detector, baseline=None, limit=None, on_progress=None,
               detector_key='research_candidate'):
    inputs, identities, group_ids = prepare_manifest(document, root, limit)
    previous = validate_baseline(baseline, document) if baseline is not None else {}
    threshold = float(detector.threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Candidate threshold must be finite and in (0,1)')
    window_samples = getattr(detector, 'window_samples', 4*RATE)
    if isinstance(window_samples, bool) or not isinstance(window_samples, int) or window_samples < 1:
        raise ValueError('Candidate window_samples must be a positive integer')
    rows = []
    for index, (case, channel, path, expected_hash) in enumerate(inputs, 1):
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != expected_hash:
            raise ValueError('V4 source changed after preflight')
        audio, rate, source = read_audio(raw)
        x, metrics, warnings = diagnostics(audio, rate)
        spans = detector.spans(x) if hasattr(detector, 'spans') else anchored_spans(len(x), window_samples)
        if spans != anchored_spans(len(x), window_samples):
            raise ValueError('Candidate must use complete anchored native-window coverage')
        windows = []
        for start, end in spans:
            if not 0 <= start < end <= len(x):
                raise ValueError('Invalid candidate window bounds')
            score = detector.score(x[start:end])
            windows.append({'start': start/RATE, 'end': end/RATE, 'aggregation_weight': 1.,
                            'analysis_samples_sha256': sha256(x[start:end].astype('<f4').tobytes()).hexdigest(),
                            **score})
        assessment = decision(windows, metrics, len(x)/RATE, threshold,
                              abstention_band(threshold, getattr(detector, 'band_logit', None)))
        assessment['aggregation'] = f'Arithmetic mean of anchored {window_samples/RATE:g}-second window scores; overlapping end-tail window has equal weight'
        group = group_ids['v4-regression:'+case['id']]
        row = {k: case[k] for k in ('id', 'label', 'generator', 'speaker', 'source', 'source_utt')}
        row.update({'id': case['id']+'::'+channel, 'source_id': case['id'], 'condition': channel,
                    'group': group, 'path': str(path.relative_to(Path(root).resolve())),
                    'source_sha256': expected_hash, 'original_sha256': case['sha256'],
                    'ref_utt': case.get('ref_utt'), 'source_header': source,
                    'analysis_samples_sha256': sha256(x.astype('<f4').tobytes()).hexdigest(),
                    'duration_seconds': len(x)/RATE, 'synthetic_score': assessment['synthetic_score'],
                    'assessment': assessment, 'diagnostics': metrics, 'warnings': warnings, 'windows': windows})
        if previous:
            old = previous[(case['id'], channel)]
            if old['label'] != case['label']:
                raise ValueError('Historical baseline label differs from manifest')
            row['historical_baseline'] = {'synthetic_score': old['score'],
                'binary_synthetic': old['score'] >= baseline['threshold'],
                'assessment_under_original_policy': old['assessment']}
        rows.append(row)
        if on_progress:
            on_progress(index, len(inputs))
    result = {'schema_version': 'core-detector-screen-1.0',
        'mode': 'supplied development authenticity labels', 'detector_key': detector_key,
        'model': detector.manifest, 'threshold': threshold, 'policy': POLICY,
        'manifest_self_digest': document['sha256'], 'manifest_original_count': len(document['cases']),
        'recording_window_policy': {'window_samples': window_samples, 'sample_rate': RATE,
            'aggregation': 'Equal arithmetic mean, complete native-size windows plus end-anchored overlapping tail',
            'padding': 'No padding by regression runner; adapter-specific native preprocessing retained',
            'comparison': 'Identical source files across systems; candidate native context can differ from the historical fusion 4-second context'},
        'original_clip_count': len(identities), 'case_count': len(rows), 'source_groups': len(set(group_ids.values())),
        'bounds': {'debug_limit_originals': limit, 'complete_manifest_run': limit is None,
                   'conditions': list(CHANNELS)},
        'input_identities': identities, 'results': rows, 'by_group': summarize(rows, threshold),
        'by_condition': {ch: totals([r for r in rows if r['condition'] == ch], threshold) for ch in CHANNELS},
        'by_condition_generator': {ch: {g: totals([r for r in rows if r['condition'] == ch and r['generator'] == g], threshold)
            for g in sorted({r['generator'] for r in rows}) if any(r['condition'] == ch and r['generator'] == g for r in rows)} for ch in CHANNELS},
        'by_condition_genuine_source': {ch: {s: totals([r for r in rows if r['condition'] == ch and r['label'] == 'human' and r['source'] == s], threshold)
            for s in sorted({r['source'] for r in rows if r['label'] == 'human'})} for ch in CHANNELS},
        'scope': 'Spent v4 English read-speech regression, not locked confirmation or a world-class claim. '
                 'No fitting, threshold tuning or product changes. All stored conditions and references remain connected to their original speaker/source groups. '
                 'Current quality and engineering abstention policy applied; no product perturbation-stability reruns. '
                 'Candidate scores complete recordings using anchored native-size windows; no centre-only screen crop. '
                 'The historical fusion uses 4-second windows; this is a source-file system comparison, not identical model inputs.'}
    if previous:
        baseline_rows = [{**r, 'synthetic_score': r['historical_baseline']['synthetic_score'],
                          'assessment': {'assessment': r['historical_baseline']['assessment_under_original_policy']}}
                         for r in rows]
        result['historical_baseline'] = {'threshold': baseline['threshold'],
            'by_condition': {ch: totals([r for r in baseline_rows if r['condition'] == ch], baseline['threshold']) for ch in CHANNELS},
            'scope': 'Stored fusion scores reused without inference. Raw comparisons use each detector frozen threshold; historical assessments retain their original policy and are not directly comparable to current candidate abstention.'}
    return result


def write_receipt(result, output, directory):
    """Publish a new output and content-addressed prior-use record without overwrite."""
    output, directory = Path(output), Path(directory)
    if output.exists():
        raise ValueError('Output already exists; choose a new filename')
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/(sha256(text.rstrip('\n').encode()).hexdigest()+'.json')
    if receipt.exists():
        if receipt.read_text() != text:
            raise ValueError('Prior-use receipt content mismatch')
    else:
        with receipt.open('x') as stream:
            stream.write(text)
    with output.open('x') as stream:
        stream.write(text)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector', choices=('pella-current', 'wavlm-aasist'), required=True,
                        help='Isolated research adapter; never changes the app detector')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--limit', type=int, help='Debug only: first N originals in frozen manifest order, all four conditions')
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error('threads must be 1..4')
    try:
        if args.output.exists():
            raise ValueError('Output already exists; choose a new filename')
        manifest_path = ROOT/'data/evaluation/v4-manifest.json'
        baseline_path = ROOT/'docs/detector-evaluation.json'
        manifest_raw, baseline_raw = manifest_path.read_bytes(), baseline_path.read_bytes()
        document, baseline = json.loads(manifest_raw), json.loads(baseline_raw)
        if len(document['cases']) != 320:
            raise ValueError('CLI requires the complete 320-original v4 manifest')
        # Fail before large model loading if any selected source or baseline differs.
        prepare_manifest(document, ROOT, args.limit)
        validate_baseline(baseline, document)
        import torch
        torch.set_num_threads(args.threads)
        started = time.perf_counter()
        if args.detector == 'pella-current':
            from lab.research_pella_current import PellaCurrentDetector
            model = PellaCurrentDetector()
            adapter_path = 'lab/research_pella_current.py'
        else:
            from lab.research_wavlm_aasist import WavLMAASISTDetector
            model = WavLMAASISTDetector()
            adapter_path = 'lab/research_wavlm_aasist.py'
        load_seconds = time.perf_counter()-started
        result = regression(document, ROOT, model, baseline, args.limit,
                            lambda n, total: print(f'Regression {n}/{total}', flush=True) if n % 40 == 0 or n == total else None,
                            detector_key=args.detector)
        import resource
        result.update({'manifest_file_sha256': sha256(manifest_raw).hexdigest(),
            'historical_baseline_file_sha256': sha256(baseline_raw).hexdigest(),
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'threads': torch.get_num_threads(), 'platform': platform.platform()},
            'runtime_including_model_load_seconds': time.perf_counter()-started,
            'model_construction_and_verified_load_seconds': load_seconds,
            'process_peak_resident_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform == 'darwin' else 1024**2),
            'code_sha256': {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('scripts/screen_candidate_regression.py', adapter_path, 'lab/forensics.py', 'lab/core_screen.py', 'lab/corpus_inventory.py')}})
        if getattr(model, 'timings', None):
            timings = model.timings
            seconds = [t['seconds'] for t in timings]
            result['score_resources'] = {'calls': len(timings), 'seconds_total': sum(seconds),
                'latency_p50_seconds': float(np.quantile(seconds, .5)), 'latency_p95_seconds': float(np.quantile(seconds, .95)),
                'realtime_factor_window_inputs': sum(seconds)/sum(t['input_seconds'] for t in timings),
                'scope': 'Sequential score calls including first call; excludes shared decode, quality checks, and model load. Overlapping windows repeat audio context.'}
        receipt = write_receipt(result, args.output, ROOT/'data/evaluation/method-runs')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f'Candidate regression failed: {exc}\n')
    print('Prior-use receipt:', receipt.relative_to(ROOT))
    for condition, counts in result['by_condition'].items():
        print(condition, counts)


if __name__ == '__main__':
    main()
