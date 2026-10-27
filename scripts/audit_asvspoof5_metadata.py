"""Audit the small official protocol archive without acquiring ASVspoof audio."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.asvspoof5_metadata import audit_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads((ROOT/"docs/research-acquisition-v1.json").read_text())["asvspoof5_metadata"]
    path = ROOT/"data/research-corpora/asvspoof5"/source["name"]
    try:
        result = audit_archive(path, source["publisher_md5"])
        result["audit_code_sha256"] = sha256((ROOT/"lab/asvspoof5_metadata.py").read_bytes()).hexdigest()
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Metadata audit could not complete: {exc}\n")
    for name, summary in result["protocols"].items():
        print(name, "rows", summary["rows"], "speaker groups by class", summary["speaker_groups_by_class"])
    print("ASVspoof audio downloaded: NO; corpus qualified: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
