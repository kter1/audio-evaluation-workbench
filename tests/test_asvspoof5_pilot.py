from io import BytesIO
import random
import tarfile

import pytest

from scripts import acquire_asvspoof5_pilot as pilot


class Response(BytesIO):
    status = 206
    def __init__(self, raw, content_range):
        super().__init__(raw)
        self.headers = {'Content-Range': content_range}
    def geturl(self):
        return 'https://zenodo.org/records/14498691/files/flac_E_aa.tar'


def test_range_budget_and_exact_publisher_response_are_required(monkeypatch):
    monkeypatch.setattr(pilot, 'PREFIX_BYTES', 8)
    url = 'https://zenodo.org/archive'
    assert pilot.prefix(url, 100, lambda *a, **k: Response(b'12345678', 'bytes 0-7/100')) == b'12345678'
    for data, bounds in [(b'123456789', 'bytes 0-7/100'), (b'123', 'bytes 0-7/100'),
                         (b'12345678', 'bytes 0-7/99')]:
        with pytest.raises(ValueError):
            pilot.prefix(url, 100, lambda *a, **k: Response(data, bounds))


def test_tar_prefix_keeps_only_complete_safe_regular_audio():
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, value in [('flac_E_eval/E_0000000001.flac', b'flac'),
                            ('../unsafe.flac', b'bad'),
                            ('flac_E_eval/E_0000000002.flac', b'x'*1000)]:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, BytesIO(value))
    found = list(pilot.members(stream.getvalue()[:2800]))
    assert [(name, data) for name,data,_ in found] == [('flac_E_eval/E_0000000001.flac', b'flac')]


def test_score_blind_selection_is_order_invariant_and_groups_never_cross():
    entries = [{'id':str(i), 'speaker':'speaker'+str(i), 'label':label}
               for label, indices in [('human', range(40)), ('synthetic', range(40,80))]
               for i in indices]
    first = pilot.choose(entries)
    random.Random(42).shuffle(entries)
    assert pilot.choose(entries) == first
    assert len(first) == len({r['speaker'] for r in first}) == 40
    assert sum(r['label']=='human' for r in first) == 20
    groups = {p:{r['speaker'] for r in first if r['partition']==p}
              for p in ('calibration_reserved', 'evaluation_reserved')}
    assert not groups['calibration_reserved'] & groups['evaluation_reserved']
    with pytest.raises(ValueError, match='20 distinct'):
        pilot.choose(entries[:10])
