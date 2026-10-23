"""Validate two complete reviews and produce a development adjudication queue."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.review_agreement import compare_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_reviews(args.packet, args.left, args.right)
        result['code_sha256'] = {name: sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ('lab/review_agreement.py', 'lab/research_metrics.py')}
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Review comparison failed: {exc}\n')
    print(f"Validated {result['case_count']} paired cases; {result['adjudication_count']} require adjudication.")


if __name__ == '__main__':
    main()
