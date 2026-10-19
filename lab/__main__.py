"""Documented CLI for dataset setup, blind handoffs, and reproducible exports."""
import argparse
import json
from pathlib import Path
import sys

from .dataset import fetch_metadata, import_asvspoof
from .demo import create_demo
from .learning import prepare_examples
from .handoff import export_package, export_results, import_results, load_package
from .reporting import adjudications_csv, annotation_csv, cases_csv, findings, history_csv, manifest_export, report_html
from .store import PHASES, Store, register_current_guide


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audio Review Lab — local annotation and QA")
    parser.add_argument("--db", default="data/demo.sqlite", help="SQLite database (default: data/demo.sqlite)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Create explicitly labeled non-speech fixtures")
    learn = commands.add_parser("prepare-learning", help="Prepare four reference examples outside the blind study")
    learn.add_argument("--archive", default="data/asvspoof/ASVspoof2021_LA_eval.tar.gz")
    fetch = commands.add_parser("fetch-metadata", help="Download official metadata, scores, mapping, and notices only")
    fetch.add_argument("--directory", default="data/asvspoof")
    fetch.add_argument("--keys-archive")
    fetch.add_argument("--maps-archive")
    importer = commands.add_parser("import-asvspoof")
    importer.add_argument("--metadata-dir", default="data/asvspoof")
    importer.add_argument("--audio-dir", required=True, help="Directory directly containing official LA_E_*.flac files")
    setup = commands.add_parser("prepare-study")
    setup.add_argument("--reviewer", default="primary")
    setup.add_argument("--seed", type=int, default=20261019)
    setup.add_argument("--counts", type=int, nargs=4, default=[20, 80, 100, 100], metavar=("CAL", "MAIN1", "MAIN2", "FRESH"))
    challenge = commands.add_parser("prepare-challenge", help="Create a blind human-vs-detector listening challenge in a new database")
    challenge.add_argument("--reviewer", required=True)
    challenge.add_argument("--manifest", default="data/evaluation/v4-manifest.json")
    challenge.add_argument("--receipt", default="docs/detector-evaluation.json")
    guide = commands.add_parser("guideline")
    guide.add_argument("--version", required=True)
    guide.add_argument("--file", required=True)
    guide.add_argument("--phase", choices=list(PHASES))
    repeats = commands.add_parser("schedule-repeats")
    repeats.add_argument("--phase", default="main-01")
    repeats.add_argument("--count", type=int, default=20)
    reveal = commands.add_parser("reveal")
    reveal.add_argument("--phase", required=True)
    package = commands.add_parser("export-package")
    package.add_argument("--reviewer", required=True)
    package.add_argument("--phase", required=True)
    package.add_argument("--count", type=int, required=True)
    package.add_argument("--output", required=True)
    load = commands.add_parser("load-package")
    load.add_argument("--file", required=True)
    results = commands.add_parser("export-results")
    results.add_argument("--output", required=True)
    returned = commands.add_parser("import-results")
    returned.add_argument("--file", required=True)
    export = commands.add_parser("export")
    export.add_argument("--directory", default="exports")
    commands.add_parser("status")
    args = parser.parse_args(argv)
    store = None
    try:
        if args.command == "demo":
            print(create_demo(args.db))
            return 0
        if args.command == "fetch-metadata":
            print(json.dumps(fetch_metadata(args.directory, args.keys_archive, args.maps_archive), indent=2))
            return 0
        store = Store(args.db)
        if args.command == "prepare-learning":
            print(f"Prepared {len(prepare_examples(store, args.archive))} reference examples outside assigned study source groups.")
        elif args.command == "prepare-challenge":
            from .challenge import prepare_challenge
            study = prepare_challenge(store, json.loads(Path(args.manifest).read_text()), json.loads(Path(args.receipt).read_text()), args.reviewer)
            print(f"Prepared {study['counts']['challenge']} blind challenge clips for {args.reviewer}. Labels and detector scores stay hidden until reveal.")
        elif args.command == "import-asvspoof":
            d = Path(args.metadata_dir)
            register_current_guide(store)
            print(json.dumps(import_asvspoof(store, d / "trial_metadata.txt", d / "lfcc_gmm_scores.txt", d / "source_mapping.tsv", args.audio_dir), indent=2))
        elif args.command == "prepare-study":
            manifest = store.prepare_study(args.reviewer, args.seed, tuple(args.counts))
            print(json.dumps({"selected": len(manifest["selected"]), "seed": manifest["seed"], "counts": manifest["counts"]}, indent=2))
        elif args.command == "guideline":
            store.add_guideline(args.version, Path(args.file).read_text(), args.phase)
            print("Guideline registered; existing versions and started phases remain immutable.")
        elif args.command == "schedule-repeats":
            print(f"Scheduled {store.schedule_repeats(args.phase, args.count)} blind repeats; each opens after 48 hours.")
        elif args.command == "reveal":
            if store.setting("reviewer_mode"):
                raise ValueError("Independent reviewer databases do not contain reference labels")
            store.reveal_phase(args.phase)
            print("Phase revealed. Original judgments remain frozen.")
        elif args.command == "export-package":
            data = export_package(store, args.reviewer, args.phase, args.count)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            print(f"Blind package saved to {output}")
        elif args.command == "load-package":
            manifest = load_package(store, Path(args.file).read_bytes())
            print(f"Loaded {len(manifest['assignments'])} blind assignments for {manifest['reviewer']}")
        elif args.command == "export-results":
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(export_results(store))
            print(f"Frozen independent reviews saved to {output}")
        elif args.command == "import-results":
            print(f"Imported {import_results(store, Path(args.file).read_bytes())} independent reviews")
        elif args.command == "export":
            if store.setting("reviewer_mode"):
                raise ValueError("Use export-results for a blind independent reviewer database")
            d = Path(args.directory)
            d.mkdir(parents=True, exist_ok=True)
            (d / "annotations.csv").write_bytes(annotation_csv(store))
            (d / "revision_history.csv").write_bytes(history_csv(store))
            (d / "cases.csv").write_bytes(cases_csv(store))
            (d / "adjudications.csv").write_bytes(adjudications_csv(store))
            (d / "metrics.json").write_text(json.dumps(findings(store), indent=2) + "\n")
            (d / "findings.html").write_text(report_html(store))
            (d / "study_manifest.json").write_text(json.dumps(manifest_export(store), indent=2) + "\n")
            print(f"Research exports saved to {d}; only revealed cases appear in annotations and findings.")
        elif args.command == "status":
            print(json.dumps(findings(store), indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
