from copy import deepcopy

import pytest

from lab.research_vad import silero_frames, summarize_speech


def metric(tp, fp, fn, tn):
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn}


def test_speech_summary_uses_only_common_annotated_cases_and_exact_denominators():
    cases = [{"evaluation_group": "one", "analyzed_duration_seconds": 1,
              "speech_methods": {"energy_activity_diagnostic": metric(20, 10, 0, 20),
                                 "webrtc": {"status": "available", "metrics": metric(18, 2, 2, 28)}}}]
    unavailable = deepcopy(cases[0])
    unavailable["speech_methods"]["webrtc"] = {"status": "unavailable"}
    result = summarize_speech({"results": cases+[unavailable]})
    assert result["case_count"] == 1
    assert result["analyzed_seconds"] == 1
    assert result["methods"]["webrtc"]["nonspeech_false_positive_rate"] == pytest.approx(2/30)
    assert result["methods"]["webrtc"]["recall"] == .9


def test_silero_refuses_unverified_model_before_importing_or_executing_it(tmp_path):
    import numpy as np
    path = tmp_path/"untrusted.jit"
    path.write_bytes(b"unverified model")
    with pytest.raises(ValueError, match="pinned"):
        silero_frames(np.zeros(16000, dtype=np.float32), path)


def test_speech_comparison_rejects_different_reference_coverage():
    case = {"evaluation_group": "one", "analyzed_duration_seconds": 1,
            "speech_methods": {"energy_activity_diagnostic": metric(20, 10, 0, 20),
                               "webrtc": {"status": "available", "metrics": metric(10, 0, 0, 10)}}}
    with pytest.raises(ValueError, match="identical reference"):
        summarize_speech({"results": [case]})


def test_reusable_silero_resets_state_on_every_recording():
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from lab.research_vad import SileroSession

    class StatefulModel:
        def __init__(self):
            self.resets = 0
            self.calls = 0

        def reset_states(self):
            self.resets += 1
            self.calls = 0

        def __call__(self, chunk, rate):
            self.calls += 1
            return np.float32(min(self.calls/100, 1))

    session = SileroSession.__new__(SileroSession)
    session.torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda x: x, __version__='test')
    session.model = StatefulModel()
    session.source = {'revision': 'test'}
    session.model_sha256 = 'test'
    a = session.predict(np.zeros(16000, dtype=np.float32))
    session.predict(np.zeros(32000, dtype=np.float32))
    b = session.predict(np.zeros(16000, dtype=np.float32))
    assert a == b
    assert session.model.resets == 3
