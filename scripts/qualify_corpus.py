"""Check prospective corpus identity, provenance, history and sample-support gates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.corpus_qualification import qualify_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prior-use", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = qualify_file(args.manifest, ROOT, args.prior_use)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Qualification could not complete: {exc}\n")
    print("Structural qualification:", "PASS" if result["structural_gates_passed"] else "BLOCKED")
    for error in result["errors"]:
        print("-", error)
    return 0 if result["structural_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
