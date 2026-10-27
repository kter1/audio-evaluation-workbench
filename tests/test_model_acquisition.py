from hashlib import sha256

import pytest

from lab.model_acquisition import download_checkpoint
from tests.test_research_acquisition import Response


URL = 'https://huggingface.co/research/model/resolve/pinned/model.safetensors'


def test_streaming_checkpoint_is_authenticated_and_cache_is_verified(tmp_path):
    raw = b'pinned checkpoint tensors'
    path = tmp_path/'model'
    options = dict(expected_size=len(raw), expected_sha256=sha256(raw).hexdigest())
    result = download_checkpoint(URL, path, **options, opener=lambda *a, **k: Response(raw, len(raw), URL))
    assert result['bytes'] == len(raw) and path.read_bytes() == raw
    def forbidden(*args, **kwargs):
        raise AssertionError('Cache must be verified without network')
    assert download_checkpoint(URL, path, **options, opener=forbidden) == result
    path.write_bytes(b'x'*len(raw))
    with pytest.raises(ValueError, match='Cached checkpoint'):
        download_checkpoint(URL, path, **options, opener=forbidden)


@pytest.mark.parametrize('raw,declared,digest,error', [
    (b'12345', 100, sha256(b'12345').hexdigest(), 'Declared'),
    (b'123456', None, sha256(b'12345').hexdigest(), 'exceeds'),
    (b'1234', None, sha256(b'12345').hexdigest(), 'mismatch'),
    (b'12345', 5, '0'*64, 'mismatch'),
])
def test_failed_checkpoint_download_cannot_publish_or_leave_debris(tmp_path, raw, declared, digest, error):
    path = tmp_path/'model'
    with pytest.raises(ValueError, match=error):
        download_checkpoint(URL, path, expected_size=5, expected_sha256=digest,
                            opener=lambda *a, **k: Response(raw, declared, URL))
    assert not path.exists() and not list(tmp_path.glob('.checkpoint-*'))
