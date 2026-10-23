"""Inventory local corpora and connected identity groups without downloading data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.corpus_inventory import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write to a new JSON file")
    parser.add_argument("--registry-output", type=Path, help="Write a new prior-use identity registry for corpus qualification")
    args = parser.parse_args()
    try:
        result = inventory(ROOT, include_records=bool(args.registry_output))
        records = result.pop("identity_records", None)
        if args.registry_output:
            if not result["inventory_complete"]:
                raise ValueError("Cannot issue prior-use registry from incomplete inventory")
            registry = {"schema_version": "prior-use-registry-1.0", "records": records,
                        "manifest_file_sha256": {m["path"]: m["file_sha256"] for m in result["manifests"]},
                        "mapping_sha256": result["mapping_sha256"],
                        "method_run_file_sha256": result["method_run_file_sha256"],
                        "scope": "known manifested historical identities plus local method-comparison receipts; no perceptual duplicate or upstream-training clearance"}
            with args.registry_output.open("x") as stream:
                stream.write(json.dumps(registry, indent=2, allow_nan=False) + "\n")
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Inventory could not complete: {exc}\n")
    print(f"Historical case rows: {result['historical_case_rows']}; unique byte hashes: {result['unique_original_byte_hashes']}")
    print(f"Connected identity groups: {result['connected_identity_groups']}")
    print(f"Groups spanning historical splits within a manifest: {len(result['groups_spanning_splits_within_a_manifest'])}")
    print("Eligible new locked groups: 0 (none independently reserved/qualified)")
    for error in result["errors"]:
        print("ERROR:", error)
    return 0 if result["inventory_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
