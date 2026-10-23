"""Acquire one pinned compact classifier for isolated research; no app changes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.research_downloads import download


def main():
    spec = json.loads((ROOT/'docs/rawnetlite-acquisition-v1.json').read_text())
    if sum(f['size'] for f in spec['files']) > spec['max_total_bytes']:
        raise ValueError('RawNetLite acquisition exceeds byte budget')
    directory = ROOT/'data/research-tools/rawnetlite'/spec['revision']
    receipts = [download(f['url'], directory/f['name'], maximum=f['size'],
                         expected_size=f['size'], git_blob_sha1=f['git_blob_sha1'])
                for f in spec['files']]
    text = json.dumps({'spec': spec, 'files': receipts,
                      'scope': 'Research only; downloads are never imported automatically'}, indent=2)+'\n'
    path = directory/'acquisition.json'
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing acquisition differs')
    if not path.exists():
        with path.open('x') as stream:
            stream.write(text)
    print('Acquired', sum(r['bytes'] for r in receipts), 'verified bytes for research.')


if __name__ == '__main__':
    main()
