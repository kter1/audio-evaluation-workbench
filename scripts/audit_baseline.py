"""Audit frozen v4 data, archived reports and metrics without downloads or inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.baseline_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new audit JSON (never overwrite an existing receipt)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    try:
        result = audit(ROOT, args.bootstrap)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Audit could not complete: {exc}\n")
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Integrity:", "PASS" if result["integrity_passed"] else "FAIL")
    print("Archived rows replayed:", result["archived_replay"]["rows"])
    for channel, item in result["evaluation"]["channels"].items():
        s = item["summary"]
        print(f"{channel}: false positives {s['false_positive']}/{s['human']}; "
              f"synthetic detected {s['synthetic'] - s['false_negative']}/{s['synthetic']}")
    for error in result["errors"]:
        print("ERROR:", error)
    print("Locked-test eligibility: NO (previously published)")
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
