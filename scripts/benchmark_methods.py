"""Compare experimental analysis methods on fixtures or annotated DEVELOPMENT audio."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.research_benchmark import analytic_cases, load_manifest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures", action="store_true")
    mode.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--silero-model", type=Path, help="Explicit pinned research model; never downloaded automatically")
    args = parser.parse_args()
    try:
        cases = analytic_cases() if args.fixtures else load_manifest(args.manifest, ROOT)
        result = run(cases, repetitions=args.bootstrap, silero_model=args.silero_model)
        from lab.research_vad import summarize_speech
        result["speech_comparison"] = summarize_speech(result)
        if args.manifest:
            document = json.loads(args.manifest.read_text())
            if "stress_protocol" in document:
                from lab.vad_stress import summarize_stress
                result["stress_comparison"] = summarize_stress(result, document)
                result["code_sha256"]["lab/vad_stress.py"] = sha256((ROOT/"lab/vad_stress.py").read_bytes()).hexdigest()
                if document["stress_protocol"]["version"] == "vad-channel-stress-1.0":
                    result["code_sha256"]["lab/vad_channel_stress.py"] = sha256((ROOT/"lab/vad_channel_stress.py").read_bytes()).hexdigest()
        result["mode"] = "analytic non-speech fixtures" if args.fixtures else "supplied development annotations"
        result["manifest_sha256"] = sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None
        if args.manifest:
            # A comparison consumes these development identities even if an
            # optional user-selected output path later fails to write.
            text = json.dumps(result, indent=2, allow_nan=False)
            directory = ROOT / "data/evaluation/method-runs"
            directory.mkdir(parents=True, exist_ok=True)
            receipt = directory / (sha256(text.encode()).hexdigest() + ".json")
            try:
                with receipt.open("x") as stream:
                    stream.write(text + "\n")
            except FileExistsError:
                if receipt.read_text() != text + "\n":
                    raise ValueError("Existing comparison receipt differs from its content hash")
            print("Prior-use receipt:", receipt.relative_to(ROOT))
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Benchmark could not complete: {exc}\n")
    print(f"Cases: {result['case_count']}; source groups: {result['source_groups']}")
    for method, metrics in result["event_aggregate"].items():
        print(f"{method}: TP {metrics['true_positive']}, FP {metrics['false_positive']}, FN {metrics['false_negative']}; F1 {metrics['f1']}")
    print("Promotion: NOT ELIGIBLE (development evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
