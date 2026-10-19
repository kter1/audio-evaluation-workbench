"""Analyze one local speech recording without the web UI."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.detectors import DETECTORS, load
from lab.forensics import MAX_BYTES, analyze, report_html, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--detector', choices=list(DETECTORS), default='fusion')
    parser.add_argument('--robustness', action='store_true', help='Run three sensitivity probes')
    parser.add_argument('--output-dir', type=Path, default=Path('data/analyses'))
    args = parser.parse_args()
    if args.audio.stat().st_size > MAX_BYTES:
        parser.error('Input exceeds 50 MiB')
    report, _ = analyze(args.audio.read_bytes(), load(args.detector), robustness=args.robustness)
    path = save_report(report, args.output_dir)
    path.with_suffix('.html').write_text(report_html(report))
    print(json.dumps(report['assessment'], indent=2))
    print('Report:', path)


if __name__ == '__main__':
    main()
