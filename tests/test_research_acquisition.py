from hashlib import sha1, sha256
from io import BytesIO
import json
from urllib.request import Request

import pytest

from lab.asvspoof5_metadata import summarize_protocol
from lab.research_downloads import ApprovedRedirects, allowed_url, download
from lab.vad_corpus import parse_binary_intervals


class Response(BytesIO):
    def __init__(self, raw, declared=None, url="https://raw.githubusercontent.com/owner/repo/file"):
        super().__init__(raw)
        self.headers = {} if declared is None else {"Content-Length": str(declared)}
        self.url = url
    def geturl(self):
        return self.url


def test_download_verifies_blob_and_cache_without_overwriting(tmp_path):
    raw = b"verified source content"
    digest = sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    target = tmp_path/"source"
    receipt = download("https://raw.githubusercontent.com/owner/repo/file", target,
                       maximum=100, expected_size=len(raw), git_blob_sha1=digest,
                       opener=lambda *a, **k: Response(raw))
    assert receipt["sha256"] == sha256(raw).hexdigest()
    def forbidden(*a, **k):
        raise AssertionError("A verified cached download must not make a request")
    assert download(receipt["url"], target, maximum=100, git_blob_sha1=digest, opener=forbidden) == receipt
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Git blob"):
        download(receipt["url"], target, maximum=100, git_blob_sha1=digest, opener=forbidden)


@pytest.mark.parametrize("response,maximum,error", [
    (lambda: Response(b"123456", declared=1000), 5, "Declared"),
    (lambda: Response(b"123456"), 5, "Streaming"),
    (lambda: Response(b""), 5, "empty"),
    (lambda: Response(b"abc", url="http://127.0.0.1/private"), 5, "approved"),
])
def test_failed_downloads_leave_no_final_file_or_temporary_debris(tmp_path, response, maximum, error):
    target = tmp_path/"never-published"
    with pytest.raises(ValueError, match=error):
        download("https://zenodo.org/record/file", target, maximum=maximum, opener=lambda *a, **k: response())
    assert not target.exists()
    assert not list(tmp_path.glob(".research-*"))


def test_redirect_is_rejected_before_following_an_unapproved_destination():
    with pytest.raises(ValueError, match="approved"):
        ApprovedRedirects().redirect_request(Request("https://zenodo.org/file"), None, 302, "Found", {}, "https://example.org/executable")
    for url in ("http://zenodo.org/file", "https://user:password@zenodo.org/file", "https://zenodo.org:8080/file"):
        with pytest.raises(ValueError):
            allowed_url(url)


def test_contiguous_binary_annotations_preserve_positive_and_negative_labels():
    text = "clip,0.000,0.100,0,0.100,0.200,1,0.200,0.300,1,0.300,0.400,0\n"
    assert parse_binary_intervals(text, "clip", .4) == [{"start": .1, "end": .3}]
    assert parse_binary_intervals("clip,0,.4,0", "clip", .4) == []
    for invalid in ("clip", "wrong,0,.4,1", "clip,0,.2,0,.3,.4,1", "clip,0,.4,2", "clip,0,.3,1"):
        with pytest.raises(ValueError):
            parse_binary_intervals(invalid, "clip", .4)


def test_protocol_grouping_connects_codec_parents_without_counting_variants_as_speakers():
    lines = [b"E_0001 E_0000000001 F - - - - bonafide bonafide -\n",
             b"E_0002 E_0000000002 M C01 1 E_0000000001 AC1 A17 spoof -\n",
             b"E_0002 E_0000000003 M C02 2 E_9999999999 AC1 A17 spoof -\n"]
    result = summarize_protocol(lines)
    assert result["rows"] == 3
    assert result["declared_speaker_ids"] == 2
    assert result["declared_connected_speaker_groups"] == 1
    assert result["cross_speaker_codec_links"] == 1
    assert result["unresolved_codec_seed_references"] == 1
    assert result["codec_conditions"]["C01"]["declared_speaker_groups"]["spoof"] == 1
    with pytest.raises(ValueError, match="duplicated"):
        summarize_protocol([lines[0], lines[0]])
    with pytest.raises(ValueError):
        summarize_protocol([b"incorrect row\n"])


def test_wheel_selection_requires_matching_release_and_never_uses_source_archives():
    from packaging.tags import Tag
    from scripts.setup_research_vad import select_wheel
    metadata = {"info": {"version": "2.0.14.post1"}, "urls": [
        {"filename": "webrtcvad_wheels-2.0.14.post1-cp313-cp313-macosx_11_0_arm64.whl", "packagetype": "bdist_wheel"},
        {"filename": "webrtcvad_wheels-2.0.14.post1.tar.gz", "packagetype": "sdist"}]}
    wheel = select_wheel(metadata, "webrtcvad-wheels", "2.0.14.post1", [Tag("cp313", "cp313", "macosx_11_0_arm64")])
    assert wheel["filename"].endswith(".whl")
    with pytest.raises(ValueError, match="No compatible"):
        select_wheel(metadata, "webrtcvad-wheels", "2.0.14.post1", [Tag("cp312", "cp312", "win_amd64")])
