"""Prepare a local model-blind packet for independent development annotation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.annotation_review import build_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = build_packet(args.manifest, ROOT, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Review preparation failed: {exc}\n')
    print(f"Prepared {receipt['case_count']} development clips; review remains pending.")
    print('Share reviewer/ only, subject to source licensing; retain coordinator-only/ locally.')


if __name__ == '__main__':
    main()
