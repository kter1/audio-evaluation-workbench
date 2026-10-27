"""Explicit optional Silero research inference, never loaded by the studio."""
from hashlib import sha1, sha256
import json
from pathlib import Path

import numpy as np

from lab.research_analysis import HOP, RATE, frame_power, intervals_from_mask


class SileroSession:
    """Verified reusable CPU model; state resets for every recording, not thread safe."""

    def __init__(self, model_path):
        model_path = Path(model_path).resolve()
        root = Path(__file__).resolve().parents[1]
        source = json.loads((root/"docs/research-acquisition-v1.json").read_text())["silero"]
        expected = next(f for f in source["files"] if f["name"] == "silero_vad.jit")
        raw = model_path.read_bytes()
        if len(raw) != expected["size"] or sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest() != expected["git_blob_sha1"]:
            raise ValueError("Silero model differs from the pinned, inspected upstream revision")
        import torch
        self.torch = torch
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.source = source
        self.model_sha256 = sha256(raw).hexdigest()

    def predict(self, samples):
        frame_power(samples)
        model = self.model
        torch = self.torch
        model.reset_states()
        x = np.asarray(samples, dtype=np.float32)
        scores = []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                chunk = np.zeros(512, dtype=np.float32)
                part = x[start:start+512]
                chunk[:len(part)] = part
                value = float(model(torch.from_numpy(chunk), RATE).item())
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Silero returned an invalid speech score")
                scores.append(value)
        # Project native 32 ms chunks onto the same 20 ms frame centers used for all
        # reference scoring. No tuned padding, hangover or segmentation policy.
        centers = np.arange(len(x)//HOP)*HOP + HOP//2
        mask = np.array(scores)[centers//512] >= .5
        return {"status": "available", "model_revision": self.source["revision"], "model_sha256": self.model_sha256,
                "torch_version": torch.__version__, "device": "cpu", "native_chunk_samples": 512,
                "threshold": .5, "threshold_tuned_on_this_corpus": False,
                "intervals": intervals_from_mask(mask), "native_chunk_speech_scores": scores,
                "model_only_zero_padding_samples": (-len(x)) % 512,
                "policy": "raw 32 ms model scores sampled at common 20 ms frame centers; state reset per recording; no official timestamp postprocessing"}


def silero_frames(samples, model_path):
    frame_power(samples)
    return SileroSession(model_path).predict(samples)


def summarize_speech(result):
    from lab.research_metrics import count_metrics
    methods = {name for row in result["results"] for name in row["speech_methods"]}
    included = []
    for row in result["results"]:
        if row["speech_methods"].get("energy_activity_diagnostic") is None:
            continue
        if all(name == "energy_activity_diagnostic" or row["speech_methods"].get(name, {}).get("status") == "available" for name in methods):
            denominators = set()
            for name in methods:
                metric = row["speech_methods"][name]
                if name != "energy_activity_diagnostic":
                    metric = metric["metrics"]
                denominators.add((metric["true_positive"]+metric["false_negative"],
                                  metric["false_positive"]+metric["true_negative"]))
            if len(denominators) != 1:
                raise ValueError("VAD methods must use identical reference-frame denominators")
            included.append(row)
    summary = {}
    for name in sorted(methods):
        counts = {k: 0 for k in ("true_positive", "false_positive", "false_negative", "true_negative")}
        for row in included:
            metric = row["speech_methods"][name]
            if name != "energy_activity_diagnostic":
                metric = metric["metrics"]
            for key in counts:
                counts[key] += metric[key]
        value = count_metrics(counts["true_positive"], counts["false_positive"], counts["false_negative"])
        negative = counts["false_positive"]+counts["true_negative"]
        value.update(true_negative=counts["true_negative"], frames=sum(counts.values()),
                     nonspeech_false_positive_rate=counts["false_positive"]/negative if negative else None)
        summary[name] = value
    return {"case_count": len(included), "analyzed_seconds": sum(r["analyzed_duration_seconds"] for r in included),
            "connected_source_groups": len({r["evaluation_group"] for r in included}),
            "methods": summary, "scope": "micro frame metrics on common annotated cases only; no population confidence claim"}
