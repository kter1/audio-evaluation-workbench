"""Generated non-speech fixtures. Their arbitrary classes are NEVER study evidence."""
from pathlib import Path
import numpy as np
import soundfile as sf

from .store import Store, register_current_guide


def create_demo(database, media_dir=None):
    store = Store(database)
    if store.one("SELECT 1 FROM recordings"):
        raise ValueError("Demo database already exists; choose a new path")
    media_dir = Path(media_dir or store.path.parent / "demo_audio").resolve()
    media_dir.mkdir(parents=True, exist_ok=True)
    records, scores = [], []
    rate = 16000
    for i in range(44):
        t = np.arange(rate * 3) / rate
        audio = 0.13 * np.sin(2 * np.pi * (200 + i * 17) * t)
        if i % 3 == 0:
            audio *= np.where((t > 1.1) & (t < 1.35), 0, 1)
        path = media_dir / f"fixture_{i:03d}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        label = "human" if i % 2 == 0 else "synthetic"
        records.append({"trial_id": f"DEMO_{i:03d}", "path": str(path), "source_group": f"demo-source-{i}",
                        "subset": "progress" if i < 12 else "eval", "codec": "fixture PCM", "attack": "arbitrary test label", "kind": "demo", "reference": label})
        score = (1 if label == "human" else -1) + i / 100
        if i in (14, 19, 28, 35):
            score *= -1
        scores.append({"trial_id": f"DEMO_{i:03d}", "score": score})
    register_current_guide(store)
    store.register(records, scores, {"dataset": "Generated non-speech demonstration fixtures", "baseline": "fabricated fixture scores", "warning": "All tones are generated. Human/synthetic test labels are arbitrary; never report as personal study evidence."})
    store.prepare_study("practice-reviewer", counts=(4, 8, 8, 8))
    store.close()
    return str(Path(database).resolve())
