"""Download the pinned, checksum-verified pellav2 checkpoint and backbone configuration."""
from pathlib import Path
import sys
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.pella import FILES, MODEL_DIR, PellaV2Detector, file_hash


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for source, (name, expected) in FILES.items():
        path = MODEL_DIR / name
        if not path.exists() or file_hash(path) != expected:
            temp = path.with_suffix(path.suffix + '.part')
            with requests.get(f'https://huggingface.co/{source}', stream=True, timeout=120) as response:
                response.raise_for_status()
                with temp.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
            if file_hash(temp) != expected:
                temp.unlink()
                raise ValueError(f'Downloaded file failed pinned SHA-256 verification: {name}')
            temp.replace(path)
        print('Verified:', name, flush=True)
    detector = PellaV2Detector()
    print(f'pellav2 is ready: {detector.parameters:,} parameters. Inference requires no network.')


if __name__ == '__main__':
    main()
