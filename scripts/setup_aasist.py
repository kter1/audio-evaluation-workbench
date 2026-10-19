"""Install the pinned official AASIST weights and configuration."""
import hashlib
from pathlib import Path
import sys
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.aasist import FILES, REVISION, ROOT


def main():
    for source,target in [('config/AASIST.conf','data/models/aasist/config.json'),('models/weights/AASIST.pth','data/models/aasist/model.pth')]:
        p=ROOT/target;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==FILES[target]:
            print('Verified:',target);continue
        r=requests.get(f'https://raw.githubusercontent.com/clovaai/aasist/{REVISION}/{source}',timeout=60);r.raise_for_status()
        if hashlib.sha256(r.content).hexdigest()!=FILES[target]:raise ValueError('Download checksum mismatch')
        temp=p.with_suffix('.part');temp.write_bytes(r.content);temp.replace(p)
        print('Verified:',target)
    print('AASIST is ready for offline inference.')


if __name__=='__main__':main()
