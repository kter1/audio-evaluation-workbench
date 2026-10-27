"""Explicit acquisition of one pinned WavLM+AASIST research candidate."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.model_acquisition import download_checkpoint
from lab.research_downloads import download


def main():
    spec = json.loads((ROOT/'docs/wavlm-aasist-acquisition-v1.json').read_text())
    if spec['checkpoint']['size']+sum(f['size'] for f in spec['files']) > spec['max_total_bytes']:
        raise ValueError('Candidate exceeds acquisition budget')
    directory = ROOT/'data/research-tools/wavlm-aasist'/spec['revision']
    receipts = [download(f['url'], directory/f['name'], maximum=f['size'],
                         expected_size=f['size'], git_blob_sha1=f['git_blob_sha1']) for f in spec['files']]
    entry = spec['checkpoint']
    print('Acquiring pinned research checkpoint:', entry['size'], 'bytes', flush=True)
    receipts.append(download_checkpoint(entry['url'], directory/entry['name'],
                    expected_size=entry['size'], expected_sha256=entry['sha256']))
    text = json.dumps({'spec': spec, 'files': receipts}, indent=2)+'\n'
    target = directory/'acquisition.json'
    if target.exists() and target.read_text() != text:
        raise ValueError('Existing acquisition differs')
    if not target.exists():
        with target.open('x') as stream:
            stream.write(text)
    print('Verified research candidate; app remains unchanged.')


if __name__ == '__main__':
    main()
