"""Acquire bounded, pinned development audio or ASVspoof protocol metadata only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.research_downloads import download
from lab.vad_corpus import ten_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("ten-vad-development", "asvspoof5-metadata", "silero-research-model"))
    args = parser.parse_args()
    spec = json.loads((ROOT/"docs/research-acquisition-v1.json").read_text())
    try:
        if args.dataset == "ten-vad-development":
            source = spec["ten_vad"]
            if sum(f["size"] for f in source["files"]) > source["max_total_bytes"]:
                raise ValueError("Pinned file list exceeds total acquisition budget")
            directory = ROOT/"data/research-corpora/ten-vad"/source["revision"]
            receipts = []
            for entry in source["files"]:
                if Path(entry["name"]).name != entry["name"]:
                    raise ValueError("Pinned file names must be simple basenames")
                receipt = download(entry["url"], directory/entry["name"], maximum=entry["size"],
                                   expected_size=entry["size"], git_blob_sha1=entry["git_blob_sha1"])
                receipt["path"] = str(Path(receipt["path"]).relative_to(ROOT))
                receipts.append(receipt)
            result = ten_manifest(ROOT, directory, source["revision"], receipts)
            target = ROOT/"data/evaluation/ten-vad-development-v1.json"
        elif args.dataset == "silero-research-model":
            source = spec["silero"]
            if sum(f["size"] for f in source["files"]) > source["max_total_bytes"]:
                raise ValueError("Model files exceed bounded research budget")
            directory = ROOT/"data/research-tools/silero"/source["revision"]
            receipts = []
            for entry in source["files"]:
                receipt = download(entry["url"], directory/entry["name"], maximum=entry["size"],
                                   expected_size=entry["size"], git_blob_sha1=entry["git_blob_sha1"])
                receipt["path"] = str(Path(receipt["path"]).relative_to(ROOT))
                receipts.append(receipt)
            result = {"revision": source["revision"], "files": receipts, "scope": source["purpose"]}
            target = directory/"acquisition-receipt.json"
        else:
            source = spec["asvspoof5_metadata"]
            destination = ROOT/"data/research-corpora/asvspoof5"/source["name"]
            result = download(source["url"], destination, maximum=source["max_bytes"], publisher_md5=source["publisher_md5"])
            result["path"] = str(destination.relative_to(ROOT))
            result["scope"] = "Official protocol metadata only; no audio or detector scores"
            target = destination.with_suffix(".receipt.json")
        text = json.dumps(result, indent=2, allow_nan=False)+"\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_text() != text:
                raise ValueError("Existing acquisition receipt differs; preserve it and investigate")
        else:
            with target.open("x") as stream:
                stream.write(text)
        print("Acquisition receipt:", target.relative_to(ROOT))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Acquisition could not complete: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
