"""Audio is decoded server-side; original names never reach blind players."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf


def _decode_flac_verified(blob):
    """Lossless fallback for original ASVspoof FLAC rejected by libsndfile.

    Accept only 16-bit PCM that matches STREAMINFO length/rate/channels and the
    source's embedded PCM MD5. Never conceal decoder errors or alter source files.
    """
    import imageio_ffmpeg
    import subprocess
    from hashlib import md5
    if len(blob) < 42 or blob[:4] != b"fLaC" or blob[4] & 0x7f != 0 or int.from_bytes(blob[5:8], "big") != 34:
        raise ValueError("Unsupported or malformed FLAC metadata")
    info = sf.info(BytesIO(blob))
    expected_md5 = blob[26:42]
    if info.subtype != "PCM_16" or not any(expected_md5):
        raise ValueError("Fallback requires 16-bit FLAC with an embedded PCM checksum")
    try:
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                                 "-xerror", "-err_detect", "explode", "-i", "pipe:0", "-f", "wav",
                                 "-c:a", "pcm_s16le", "pipe:1"], input=blob, capture_output=True, timeout=30, check=True)
    except (subprocess.SubprocessError, OSError) as error:
        raise ValueError("FLAC fallback rejected the recording") from error
    pcm, rate = sf.read(BytesIO(result.stdout), dtype="int16", always_2d=True)
    if (len(pcm) != info.frames or rate != info.samplerate or pcm.shape[1] != info.channels
            or md5(pcm.astype("<i2", copy=False).tobytes()).digest() != expected_md5):
        raise ValueError("Decoded FLAC does not match its source PCM checksum or format")
    return pcm.astype("float32") / 32768, rate


def checksum(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(path):
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except RuntimeError:
        if hasattr(path, "getvalue"):
            blob = path.getvalue()
        elif hasattr(path, "read"):
            path.seek(0)
            blob = path.read()
        else:
            blob = Path(path).read_bytes()
        if not blob.startswith(b"fLaC"):
            raise
        audio, rate = _decode_flac_verified(blob)
    if not len(audio) or rate <= 0 or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite audio")
    return audio, rate


def wav_bytes(audio, rate, start=0.0, end=None):
    duration = len(audio) / rate
    end = duration if end is None else end
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration:
        raise ValueError("Segment must satisfy 0 ≤ start < end ≤ duration")
    first, last = round(start * rate), round(end * rate)
    if first >= last:
        raise ValueError("Segment is shorter than one sample")
    stream = BytesIO()
    sf.write(stream, audio[first:last], rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()
