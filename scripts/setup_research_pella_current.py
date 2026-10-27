"""Explicit bounded acquisition of the newer Pella model; no app integration."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.model_acquisition import download_checkpoint
from lab.research_downloads import download


def main():
    spec = json.loads((ROOT/'docs/pella-current-acquisition-v1.json').read_text())
    if spec['checkpoint']['size']+sum(f['size'] for f in spec['files']) > spec['max_total_bytes']:
        raise ValueError('Candidate exceeds acquisition budget')
    directory = ROOT/'data/research-tools/pella-current'/spec['revision']
    receipts = []
    for entry in spec['files']:
        receipts.append(download(entry['url'], directory/entry['name'], maximum=entry['size'],
                                 expected_size=entry['size'], git_blob_sha1=entry['git_blob_sha1']))
    entry = spec['checkpoint']
    print('Acquiring pinned research checkpoint:', entry['size'], 'bytes', flush=True)
    receipts.append(download_checkpoint(entry['url'], directory/entry['name'],
                    expected_size=entry['size'], expected_sha256=entry['sha256']))
    text = json.dumps({'spec': spec, 'files': receipts}, indent=2)+'\n'
    target = directory/'acquisition-v2.json'
    if target.exists() and target.read_text() != text:
        raise ValueError('Existing acquisition differs')
    if not target.exists():
        with target.open('x') as stream:
            stream.write(text)
    print('Verified research candidate; app remains unchanged.')


if __name__ == '__main__':
    main()
