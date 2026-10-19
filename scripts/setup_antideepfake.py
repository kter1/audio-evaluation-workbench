"""Download the pinned, checksum-verified NII AntiDeepfake MMS-300M checkpoint (safetensors).

The weights are licensed CC BY-NC-SA 4.0 (research and educational use only). They are downloaded to
data/models/antideepfake and never committed.
"""
from pathlib import Path
import sys
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.antideepfake import FILES, MODEL_DIR, REPOSITORY, REVISION, AntiDeepfakeDetector, file_hash


def main():
    print('NII AntiDeepfake weights: CC BY-NC-SA 4.0, non-commercial research and educational use only.')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = MODEL_DIR / name
        if not path.exists() or file_hash(path) != expected:
            temp = path.with_suffix(path.suffix + '.part')
            with requests.get(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}', stream=True, timeout=120) as response:
                response.raise_for_status()
                with temp.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
            if file_hash(temp) != expected:
                temp.unlink()
                raise ValueError(f'Downloaded file failed pinned SHA-256 verification: {name}')
            temp.replace(path)
        print('Verified:', name, flush=True)
    detector = AntiDeepfakeDetector()
    print(f'AntiDeepfake is ready: {detector.parameters:,} parameters. Inference requires no network.')


if __name__ == '__main__':
    main()
