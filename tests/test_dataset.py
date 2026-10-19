from io import BytesIO
import json

import numpy as np
import pytest
import soundfile as sf

from lab.audio import decode, wav_bytes
from lab.dataset import parse_mapping, source_group
from lab.demo import create_demo
from lab.store import Store


def test_sample_reproducible_balanced_and_source_disjoint(db, tmp_path):
    other_path = tmp_path / "other.sqlite"
    create_demo(other_path)
    other = Store(other_path)
    assert db.study_manifest()["selected"] == other.study_manifest()["selected"]
    rows = db.rows("SELECT a.phase,r.source_group,l.label FROM assignments a JOIN recordings r ON a.recording_id=r.id JOIN reference_labels l ON l.recording_id=r.id")
    assert len({r["source_group"] for r in rows}) == len(rows)
    progress = {r["source_group"] for r in db.rows("SELECT source_group FROM recordings WHERE subset='progress'")}
    assert not progress & {r["source_group"] for r in rows}
    for phase in {r["phase"] for r in rows}:
        labels = [r["label"] for r in rows if r["phase"] == phase]
        assert labels.count("human") == labels.count("synthetic")
    before = db.one("SELECT config FROM threshold_config")["config"]
    assert db.freeze_threshold() == json.loads(before)
    with pytest.raises(ValueError, match="frozen"):
        db.prepare_study()
    other.close()


def test_precise_segment_replay_uses_sample_indices():
    rate = 16000
    audio = np.zeros((rate, 1), dtype=np.float32)
    clipped, actual_rate = decode(BytesIO(wav_bytes(audio, rate, 0.123, 0.456)))
    assert actual_rate == rate
    assert len(clipped) == round(0.456 * rate) - round(0.123 * rate)
    with pytest.raises(ValueError, match="Segment"):
        wav_bytes(audio, rate, 1, 0)


def test_flac_fallback_preserves_pcm_and_rejects_checksum_mismatch(tmp_path):
    from lab.audio import _decode_flac_verified
    import hashlib
    path = tmp_path / "source.flac"
    pcm = np.random.default_rng(12).integers(-30000, 30000, (8000, 1), dtype=np.int16)
    sf.write(path, pcm, 16000, subtype="PCM_16")
    blob = path.read_bytes()
    audio, rate = _decode_flac_verified(blob)
    assert rate == 16000
    assert np.array_equal(audio, pcm.astype("float32") / 32768)
    broken = bytearray(blob)
    broken[26] ^= 1
    with pytest.raises(ValueError):
        _decode_flac_verified(bytes(broken))


def test_source_mapping_handles_bonafide_voice_conversion_and_tts(tmp_path):
    assert source_group({"VCTK_ID": "p300_007", "VC_source_VCTK_ID": "-", "TTS_text": "-"}) == source_group({"VCTK_ID": "-", "VC_source_VCTK_ID": "p300_007", "TTS_text": "-"})
    assert source_group({"TTS_text": "Hello, WORLD!"}) == source_group({"TTS_text": " hello world "})
    assert source_group({"TTS_text": "-"}) is None
    path = tmp_path / "mapping.tsv"
    path.write_text("ASVspoof_ID\tVCTK_ID\tTTS_text\tVC_source_VCTK_ID\nLA_E_1\tp300_007\t-\t-\nLA_E_1\tp300_007\t-\t-\n")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_mapping(path)


@pytest.mark.parametrize("case", ["duplicate_recording", "duplicate_score", "unknown_score", "nan_score"])
def test_registration_errors_are_atomic(tmp_path, case):
    store = Store(tmp_path / "new.sqlite")
    record = {"trial_id": "a", "path": "/unused.flac", "source_group": "x", "subset": "eval", "codec": "none", "attack": "none", "kind": "research", "reference": "human"}
    records, scores = [record], [{"trial_id": "a", "score": 1.0}]
    if case == "duplicate_recording":
        records *= 2
    elif case == "duplicate_score":
        scores *= 2
    elif case == "unknown_score":
        scores[0]["trial_id"] = "unknown"
    else:
        scores[0]["score"] = float("nan")
    with pytest.raises(ValueError):
        store.register(records, scores, {})
    assert not store.rows("SELECT * FROM recordings")
    assert not store.rows("SELECT * FROM provenance")
    store.close()
