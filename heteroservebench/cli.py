"""Command line interface for HeteroServeBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from heteroservebench.calibration import dry_run_plan, parse_workload_ids, run_calibration
from heteroservebench.config import load_config
from heteroservebench.metrics import summarize_run
from heteroservebench.runner import run_experiment
from heteroservebench.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="heteroservebench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an experiment")
    run_parser.add_argument("--config", required=True, type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate a run directory")
    validate_parser.add_argument("--run-dir", required=True, type=Path)
    validate_parser.add_argument("--strict-scientific", action="store_true")

    calibrate_parser = subparsers.add_parser("calibrate", help="calibrate single-GPU offered-load capacity")
    calibrate_parser.add_argument("--config", required=True, type=Path)
    calibrate_parser.add_argument("--workloads", required=True)
    calibrate_parser.add_argument("--output-root", type=Path, default=Path("calibration_runs"))
    calibrate_parser.add_argument("--cooldown-s", type=float, default=5.0)
    calibrate_parser.add_argument("--dry-run", action="store_true")
    calibrate_parser.add_argument("--dry-run-l0-s", type=float, default=1.0)

    summarize_parser = subparsers.add_parser("summarize", help="summarize a run directory")
    summarize_parser.add_argument("--run-dir", required=True, type=Path)
    summarize_parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            config = load_config(args.config)
            run_dir = run_experiment(config)
            print(str(run_dir))
            return 0
        if args.command == "validate":
            report = validate_run(args.run_dir, strict_scientific=args.strict_scientific)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["valid"] else 2
        if args.command == "calibrate":
            config = load_config(args.config)
            workload_ids = parse_workload_ids(args.workloads)
            if args.dry_run:
                plan = dry_run_plan(config, workload_ids, assumed_l0_s=args.dry_run_l0_s)
                print(json.dumps(plan, indent=2, sort_keys=True))
                return 0
            calibration_dir = run_calibration(
                config,
                workload_ids,
                output_root=args.output_root,
                cooldown_s=args.cooldown_s,
            )
            print(str(calibration_dir))
            return 0
        if args.command == "summarize":
            summary = summarize_run(args.run_dir, write=not args.no_write)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    except (ValidationError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
