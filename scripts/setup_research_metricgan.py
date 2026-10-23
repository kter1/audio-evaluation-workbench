"""Acquire the pinned 7.6 MB MetricGAN research model and source for inspection."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.research_downloads import download


def main():
    spec=json.loads((ROOT/'docs/metricgan-acquisition-v1.json').read_text())
    if sum(f['size'] for f in spec['files'])+len(spec['source_files'])*256*1024>spec['max_total_bytes']:
        raise ValueError('Research acquisition exceeds budget')
    directory=ROOT/'data/research-tools/metricgan'/spec['model_revision']
    receipts=[]
    for f in spec['files']:
        receipts.append(download(f['url'],directory/f['name'],maximum=f['size'],expected_size=f['size'],git_blob_sha1=f['git_blob_sha1']))
    for path in spec['source_files']:
        url=f"https://raw.githubusercontent.com/speechbrain/speechbrain/{spec['source_revision']}/{path}"
        receipts.append(download(url,directory/'upstream'/Path(path).name,maximum=256*1024))
    if sum(r['bytes'] for r in receipts)>spec['max_total_bytes']:raise ValueError('Actual acquisition exceeds budget')
    result={'spec':spec,'files':receipts,'scope':'Research only; no installation, remote code execution or production integration'}
    target=directory/'acquisition.json';text=json.dumps(result,indent=2)+'\n'
    if target.exists() and target.read_text()!=text:raise ValueError('Existing acquisition differs')
    if not target.exists():
        with target.open('x') as stream:stream.write(text)
    print('Acquired',sum(r['bytes'] for r in receipts),'verified bytes for research.')


if __name__=='__main__':main()
