"""Acquire only the pinned 45-clip public modern-voice development screen (12.2 MB)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.baseline_audit import checked_path
from lab.research_downloads import download


def main():
    source = ROOT/'docs/core-modern-acquisition-v1.json'
    raw = source.read_bytes()
    spec = json.loads(raw)
    files = spec['files']
    if len(files) != 45 or sum(row['bytes'] for row in files) > 16*1024**2:
        raise ValueError('Acquisition exceeds fixed 45-clip / 16-MiB budget')
    for row in files:
        target = checked_path(ROOT, row['path'])
        download(row['url'], target, maximum=4*1024**2, expected_size=row['bytes'], expected_sha256=row['sha256'])
    target = ROOT/'data/research-corpora/voxenes-2026/acquisition.json'
    if target.exists() and target.read_bytes() != raw:
        raise ValueError('Existing acquisition receipt differs; do not overwrite it')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open('xb') as stream: stream.write(raw)
    print('Verified 45 public development clips; no full dataset or model download.')


if __name__ == '__main__':
    main()
