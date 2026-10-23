"""Stream an explicitly pinned research checkpoint without executing its contents.

Separate from the small-file downloader: never invoked by app startup or upload.
"""
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from lab.research_downloads import allowed_url, open_approved

MAX_CHECKPOINT_BYTES = 3*1024**3


def file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(url, destination, *, expected_size, expected_sha256,
                        opener=open_approved):
    allowed_url(url)
    if not 0 < expected_size <= MAX_CHECKPOINT_BYTES or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A bounded size and pinned SHA-256 are required')
    path = Path(destination)
    if path.exists():
        if path.stat().st_size != expected_size or file_sha256(path) != expected_sha256:
            raise ValueError('Cached checkpoint differs from pinned identity')
        return {'bytes': expected_size, 'sha256': expected_sha256, 'url': url}
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < expected_size+512*1024**2:
        raise ValueError('Insufficient free space for checkpoint acquisition')
    temporary = None
    started = time.monotonic()
    try:
        with opener(url, timeout=30) as response:
            allowed_url(response.geturl())
            declared = response.headers.get('Content-Length')
            if declared is not None and int(declared) != expected_size:
                raise ValueError('Declared checkpoint size differs from pinned metadata')
            digest = sha256()
            total = 0
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as stream:
                temporary = Path(stream.name)
                while True:
                    if time.monotonic()-started > 1200:
                        raise ValueError('Checkpoint acquisition exceeded 20-minute budget')
                    chunk = response.read(min(4*1024**2, expected_size-total+1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_size:
                        raise ValueError('Checkpoint stream exceeds pinned size')
                    digest.update(chunk)
                    stream.write(chunk)
            if total != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError('Checkpoint size or SHA-256 mismatch')
        os.link(temporary, path)  # Exclusive publication; existing evidence is never replaced.
        return {'bytes': total, 'sha256': expected_sha256, 'url': url}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
