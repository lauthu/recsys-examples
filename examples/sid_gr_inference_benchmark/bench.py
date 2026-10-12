#!/usr/bin/env python3
"""SID-GR benchmark suite: end-to-end measurements and optional diagnosis."""

import argparse
import sys
from pathlib import Path

from harness.config import ROOT, load
from harness.runner import e2e, select_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in {
        "plan": "List selected cases and protocol; no GPU work",
        "e2e": "Run candidates and save measurements on the selected GPU",
    }.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, default=ROOT / "config/runs/default.yml")
        p.add_argument("--case", help="Exact case ID or shell-style pattern")
        p.add_argument("--solutions", nargs="+")
        p.add_argument(
            "--smoke",
            action="store_true",
            help="First selected case; two warmups and three measurements",
        )
    p = sub.add_parser(
        "diagnose", help="Capture GPU kernels for a saved case with Nsight"
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--case", required=True)
    p = sub.add_parser(
        "report", help="Rebuild run, diagnosis and leaderboard reports; no GPU work"
    )
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in {"plan", "e2e"}:
            cfg = load(args.config)
            if args.solutions:
                if len(set(args.solutions)) != len(args.solutions):
                    raise ValueError("Selected solutions must be unique")
                if set(args.solutions) - set(cfg["solutions"]):
                    raise ValueError(
                        "Selected solution is not in the run configuration"
                    )
                cfg["solutions"] = args.solutions
            cfg["smoke"] = args.smoke
            selected = select_cases(cfg, args.case, args.smoke)
            if args.command == "plan":
                for case in selected:
                    print(
                        f"{case['id']}  solutions={','.join(cfg['solutions'])}  measurement={case['measurement']}"
                    )
                print(
                    f"{len(selected)} cases; {len(selected) * len(cfg['solutions'])} jobs"
                )
            else:
                _, manifest = e2e(cfg, selected)
                return int(any(j["status"] != "success" for j in manifest["jobs"]))
        elif args.command == "diagnose":
            from harness.profiling.diagnose import run

            _, records = run(args.run, args.case)
            return int(
                any(r["summary"]["status"] in {"failed", "timed_out"} for r in records)
            )
        elif args.command == "report":
            from harness.reporting.diagnosis import render as render_diagnosis
            from harness.reporting.report import render

            for path in sorted((args.run / "diagnostics").glob("*/diagnosis.json")):
                render_diagnosis(path.parent)
            render(args.run)
            print(args.run / "report.md")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
