"""Download a pinned, checksum-verified safetensors checkpoint; no model code."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.forensics import MODEL_DIR, MODEL_ID, MODEL_FILES, REVISION, verify_model

EXPECTED = MODEL_FILES


def file_hash(path):
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": MODEL_ID, "revision": REVISION, "files": {}}
    for name, expected in EXPECTED.items():
        path = MODEL_DIR / name
        if not path.exists() or file_hash(path) != expected:
            temp = path.with_suffix(path.suffix + ".part")
            with requests.get(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", stream=True, timeout=120) as response:
                response.raise_for_status()
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
            if file_hash(temp) != expected:
                raise ValueError(f"Downloaded file failed pinned SHA-256 verification: {name}")
            temp.replace(path)
        manifest["files"][name] = {"sha256": expected, "bytes": path.stat().st_size}
        print("Verified:", name, flush=True)
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_model()
    print("Detector is ready. Inference requires no network.")


if __name__ == "__main__":
    main()
