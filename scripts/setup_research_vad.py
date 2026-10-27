"""Install a tiny pinned binary VAD into an isolated research directory only."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.research_downloads import download


def select_wheel(metadata, distribution, version, compatible_tags):
    ranking = {tag: i for i, tag in enumerate(compatible_tags)}
    candidates = []
    if metadata["info"]["version"] != version:
        raise ValueError("PyPI version differs from pinned release")
    for item in metadata["urls"]:
        if item.get("packagetype") != "bdist_wheel" or item.get("yanked"):
            continue
        name, release, _, tags = parse_wheel_filename(item["filename"])
        common = set(tags) & ranking.keys()
        if common and name == canonicalize_name(distribution) and str(release) == version:
            candidates.append((min(ranking[tag] for tag in common), item["filename"], item))
    if not candidates:
        raise ValueError("No compatible prebuilt wheel; source compilation is intentionally disabled")
    return min(candidates, key=lambda x: x[:2])[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads((ROOT/"docs/research-acquisition-v1.json").read_text())["webrtc"]
    base = ROOT/"data/research-tools"/("webrtcvad-"+source["version"])
    try:
        metadata_path = base/"pypi-release.json"
        metadata_receipt = download(source["metadata_url"], metadata_path, maximum=source["max_metadata_bytes"])
        metadata = json.loads(metadata_path.read_text())
        entry = select_wheel(metadata, source["distribution"], source["version"], list(sys_tags()))
        if entry["size"] > source["max_wheel_bytes"] or Path(entry["filename"]).name != entry["filename"]:
            raise ValueError("Wheel exceeds bounded package budget or has an invalid filename")
        wheel = base/"wheels"/entry["filename"]
        wheel_receipt = download(entry["url"], wheel, maximum=source["max_wheel_bytes"],
                                 expected_size=entry["size"], expected_sha256=entry["digests"]["sha256"])
        target = ROOT/source["target"]
        lock = base/"platform-wheel.lock"
        lock_text = wheel.as_uri()+" --hash=sha256:"+entry["digests"]["sha256"]+"\n"
        if lock.exists() and lock.read_text() != lock_text:
            raise ValueError("Existing platform lock differs; use a separate research environment")
        lock.write_text(lock_text)
        if not target.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
                            "--only-binary=:all:", "--target", str(target), "-r", str(lock)], check=True)
        installed = {}
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                simple = PurePosixPath(name)
                if simple.is_absolute() or ".." in simple.parts:
                    raise ValueError("Invalid wheel member path")
                if (name == "webrtcvad.py" or (simple.name.startswith("_webrtcvad") and simple.suffix in (".so", ".pyd"))
                        or name.endswith(".dist-info/METADATA")):
                    path = target/name
                    expected = sha256(archive.read(name)).hexdigest()
                    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
                        raise ValueError("Installed VAD payload differs from the verified wheel")
                    installed[name] = expected
        if not any(n.endswith((".so", ".pyd")) for n in installed):
            raise ValueError("Native VAD payload missing")
        env = {**os.environ, "PYTHONPATH": str(target)}
        probe = subprocess.run([sys.executable, "-c",
            "import json,webrtcvad,importlib.metadata; print(json.dumps({'version':importlib.metadata.version('webrtcvad-wheels'),'silence_is_speech':webrtcvad.Vad(2).is_speech(bytes(640),16000)}))"],
            env=env, check=True, capture_output=True, text=True)
        result = json.loads(probe.stdout)
        if result["version"] != source["version"] or result["silence_is_speech"]:
            raise ValueError("Pinned VAD smoke probe failed")
        for receipt in (metadata_receipt, wheel_receipt):
            receipt["path"] = str(Path(receipt["path"]).relative_to(ROOT))
        result.update(schema_version="research-vad-runtime-1.0", distribution=source["distribution"],
                      isolated_pythonpath=str(target.relative_to(ROOT)), metadata=metadata_receipt,
                      wheel=wheel_receipt, installed_payload_sha256=installed,
                      scope="Research only; main environment and production dependencies unchanged")
        text = json.dumps(result, indent=2, allow_nan=False)+"\n"
        receipt_path = base/"runtime-receipt.json"
        if receipt_path.exists() and receipt_path.read_text() != text:
            raise ValueError("Runtime receipt differs; preserve existing evidence")
        receipt_path.write_text(text)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(text)
        print("Research PYTHONPATH:", target.relative_to(ROOT))
        print("Native payload hashes verified; silence probe passed.")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Research setup could not complete: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
