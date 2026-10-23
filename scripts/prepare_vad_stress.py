"""Prepare bounded deterministic VAD development stress variants, without downloads."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.vad_stress import build_stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--extended', action='store_true', help='Channel, codec and short-clip suite instead of gain/noise suite')
    args = parser.parse_args()
    try:
        result = build_stress(args.manifest, ROOT, args.output, extended=args.extended)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Stress preparation failed: {exc}\n')
    print(f"Prepared {len(result['cases'])} cases; inherited development timing only.")


if __name__ == '__main__':
    main()
