"""Bounded, verified research downloads. No archive extraction or code execution."""
from hashlib import md5, sha1, sha256
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

HOSTS = {"raw.githubusercontent.com", "zenodo.org", "pypi.org", "files.pythonhosted.org", "www.kaggle.com", "storage.googleapis.com", "huggingface.co", "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co"}


def allowed_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username
            or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("Research download URL must use an approved HTTPS host")


class ApprovedRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        allowed_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def open_approved(url, timeout):
    return build_opener(ApprovedRedirects()).open(url, timeout=timeout)


def verify(raw, *, maximum, expected_size=None, git_blob_sha1=None, expected_sha256=None, publisher_md5=None):
    if not raw or len(raw) > maximum:
        raise ValueError("Download is empty or exceeds its byte budget")
    if expected_size is not None and len(raw) != expected_size:
        raise ValueError("Download size differs from pinned source metadata")
    if git_blob_sha1 and sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest() != git_blob_sha1:
        raise ValueError("Git blob identity differs from pinned revision")
    if expected_sha256 and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Download SHA-256 mismatch")
    if publisher_md5 and md5(raw).hexdigest() != publisher_md5:
        raise ValueError("Publisher archive checksum mismatch")
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest(),
            "git_blob_sha1": git_blob_sha1, "publisher_md5": publisher_md5}


def download(url, destination, *, maximum, expected_size=None, git_blob_sha1=None,
             expected_sha256=None, publisher_md5=None, opener=open_approved):
    allowed_url(url)
    if maximum <= 0 or maximum > 32*1024*1024:
        raise ValueError("Per-file budget must be between 1 byte and 32 MiB")
    destination = Path(destination)
    checks = dict(maximum=maximum, expected_size=expected_size, git_blob_sha1=git_blob_sha1,
                  expected_sha256=expected_sha256, publisher_md5=publisher_md5)
    if destination.exists():
        if destination.stat().st_size > maximum:
            raise ValueError("Cached file exceeds its byte budget")
        with destination.open("rb") as stream:
            result = verify(stream.read(maximum+1), **checks)
        return {"url": url, "path": str(destination), **result}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with opener(url, timeout=30) as response:
            allowed_url(response.geturl())
            declared = response.headers.get("Content-Length")
            if declared is not None and (int(declared) > maximum or int(declared) < 1):
                raise ValueError("Declared download size exceeds its budget")
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".research-", delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while True:
                    chunk = response.read(min(65536, maximum-total+1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError("Streaming download exceeds its byte budget")
                    stream.write(chunk)
        result = verify(temporary.read_bytes(), **checks)
        # Link creates the final path exclusively; no existing evidence is replaced.
        os.link(temporary, destination)
        return {"url": url, "path": str(destination), **result}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
