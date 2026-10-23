"""Analytic source-level fixture validation; not a speech-detector benchmark."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.source_measurements import VERSION, measure_source


def validate():
    contract_path = ROOT / "docs/evaluation-contract-v1.json"
    contract = json.loads(contract_path.read_text())
    tolerance = contract["measurement"]["analytic_fixture_level_error_db_max"]
    rows = []
    for rate in (8000, 16000, 44100, 48000, 96000, 192000):
        t = np.arange(rate) / rate
        for amplitude in (.05, .2, .8):
            audio = (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[:, None]
            channel = measure_source(audio, rate)["channels"][0]
            expected_rms = 20 * np.log10(amplitude / np.sqrt(2))
            expected_peak = 20 * np.log10(amplitude)
            rows.append({"sample_rate_hz": rate, "amplitude": amplitude,
                         "rms_error_db": abs(channel["rms_dbfs"] - float(expected_rms)),
                         "sample_peak_error_db": abs(channel["sample_peak_dbfs"] - float(expected_peak)),
                         "crest_factor_error_db": abs(channel["crest_factor_db"] - float(20 * np.log10(np.sqrt(2))))})
    maximum = max(r[k] for r in rows for k in ("rms_error_db", "sample_peak_error_db", "crest_factor_error_db"))
    alternating = np.tile([1.2, -1.2], 8000).astype(np.float32)
    cancelled = measure_source(np.stack([alternating, -alternating], axis=1), 16000)
    silent = measure_source(np.zeros((16000, 2), np.float32), 16000)
    checks = {
        "analytic_levels_within_contract": maximum <= tolerance,
        "opposite_phase_source_peaks_preserved": all(abs(c["sample_peak_amplitude"] - 1.2) < 1e-6 for c in cancelled["channels"]),
        "opposite_phase_complete_cancellation_measured": cancelled["arithmetic_mixdown"]["complete_cancellation"],
        "over_full_scale_source_samples_counted": all(c["over_full_scale_samples"] == 16000 for c in cancelled["channels"]),
        "silence_has_explicit_null_levels": all(c["rms_dbfs"] is None and c["sample_peak_dbfs"] is None for c in silent["channels"]),
    }
    return {"schema_version": "source-measurement-validation-1.0", "measurement_version": VERSION,
            "contract_sha256": sha256(contract_path.read_bytes()).hexdigest(),
            "measurement_code_sha256": sha256((ROOT / "lab/source_measurements.py").read_bytes()).hexdigest(),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "sine_fixtures": rows, "maximum_level_error_db": maximum,
            "allowed_level_error_db": tolerance, "checks": checks, "passed": all(checks.values()),
            "limitations": ["Generated analytic fixtures validate measurement arithmetic only.",
                            "No natural-audio anomaly precision/recall, speech authenticity, edit localization or true-peak validation is established."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new receipt without overwriting existing evidence")
    args = parser.parse_args()
    result = validate()
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Analytic sine fixtures: {len(result['sine_fixtures'])}; maximum level error: {result['maximum_level_error_db']:.8f} dB")
    print(f"Checks: {sum(result['checks'].values())}/{len(result['checks'])} passed")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
