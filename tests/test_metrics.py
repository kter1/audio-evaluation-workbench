import math

import pytest

from lab.metrics import agreement, confusion, equal_error_threshold, human_metrics
from lab.reporting import csv_bytes


def test_hand_calculated_agreement():
    result = agreement(["human", "human", "synthetic", "uncertain"], ["human", "synthetic", "synthetic", "uncertain"])
    assert result["agreement"] == 0.75
    # Expected agreement = (2*1 + 1*2 + 1*1)/16 = 5/16.
    assert result["kappa"] == pytest.approx(7/11)
    assert agreement([], [])["agreement"] is None
    assert agreement(["human"], ["human"])["kappa"] is None


def test_uncertainty_and_decided_accuracy_have_different_denominators():
    rows = [{"judgment": "human", "reference": "human"}, {"judgment": "synthetic", "reference": "human"}, {"judgment": "uncertain", "reference": "synthetic"}]
    result = human_metrics(rows)
    assert result["accuracy_among_decided"] == 0.5
    assert result["uncertainty_rate"] == 1/3
    assert human_metrics([])["accuracy_among_decided"] is None
    assert human_metrics(rows[2:])["accuracy_among_decided"] is None


def test_detector_denominators_and_single_class():
    rows = [{"reference": "human", "prediction": "human"}, {"reference": "human", "prediction": "synthetic"}, {"reference": "synthetic", "prediction": "human"}]
    result = confusion(rows)
    assert result["genuine_rejected"] == 1 and result["genuine_total"] == 2
    assert result["genuine_rejection_rate"] == 0.5
    assert result["spoof_acceptance_rate"] == 1
    assert confusion(rows[:2])["spoof_acceptance_rate"] is None


def test_equal_error_threshold_direction_and_ties():
    rows = [{"score": -2, "reference": "synthetic"}, {"score": -1, "reference": "synthetic"}, {"score": 1, "reference": "human"}, {"score": 2, "reference": "human"}]
    result = equal_error_threshold(rows)
    assert result["threshold"] == 0
    assert result["eer_at_operating_point"] == 0
    tied = equal_error_threshold([{"score": 0, "reference": "human"}, {"score": 0, "reference": "synthetic"}])
    assert tied["eer_at_operating_point"] == 0.5
    with pytest.raises(ValueError, match="both"):
        equal_error_threshold(rows[:2])
    with pytest.raises(ValueError, match="finite"):
        equal_error_threshold([{"score": math.nan, "reference": "human"}])


def test_spreadsheet_formula_injection_and_unicode():
    text = csv_bytes([{"note": "  =HYPERLINK(\"bad\")", "text": "café, 音声\nsecond line", "score": -0.5}]).decode("utf-8-sig")
    import csv
    from io import StringIO
    row = next(csv.DictReader(StringIO(text)))
    assert row["note"].startswith("'  =")
    assert row["text"] == "café, 音声\nsecond line"
    assert row["score"] == "-0.5"
