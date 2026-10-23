"""Prepare explicit dispute decisions or export completed development annotations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.review_adjudication import export_adjudicated, prepare_adjudication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'export'))
    for name in ('packet', 'left', 'right', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.mode == 'export' and (args.decisions is None or args.manifest is None):
        parser.error('export requires --decisions and --manifest')
    try:
        if args.mode == 'prepare':
            result = prepare_adjudication(args.packet, args.left, args.right)
        else:
            result = export_adjudicated(args.packet, args.left, args.right, args.decisions, args.manifest)
            result['review_evidence']['code_sha256'] = {
                name: sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('lab/review_adjudication.py', 'lab/review_agreement.py', 'lab/research_metrics.py')}
        text = json.dumps(result, indent=2, allow_nan=False)+'\n'
        with args.output.open('x') as stream:
            stream.write(text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Adjudication failed: {exc}\n')
    print('Prepared pending decisions.' if args.mode == 'prepare' else 'Exported development annotations; no qualification or promotion implied.')


if __name__ == '__main__':
    main()
