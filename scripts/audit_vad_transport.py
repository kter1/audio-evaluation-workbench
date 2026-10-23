"""Audit existing transport quality warnings on constructed development fixtures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.vad_transport_audit import audit_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit_transport(args.manifest, ROOT)
        result['code_sha256'] = {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('lab/vad_transport_audit.py', 'lab/forensics.py', 'lab/source_measurements.py')}
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Transport audit failed: {exc}\n')
    print(f"Transport diagnostics: {result['passed_cases']}/{result['case_count']} passed")
    return 0 if result['passed_cases'] == result['case_count'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
