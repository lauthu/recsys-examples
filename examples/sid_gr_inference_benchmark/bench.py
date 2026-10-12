#!/usr/bin/env python3
"""SID-GR benchmark suite: end-to-end measurements and optional diagnosis."""

import argparse
import sys
from pathlib import Path

from config import ROOT, load
from runner import e2e, select_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["plan", "e2e"]:
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "benchmark.yml")
        p.add_argument("--case", help="Exact case ID or shell-style pattern")
        p.add_argument("--solutions", nargs="+")
        p.add_argument(
            "--smoke",
            action="store_true",
            help="First selected case; two warmups and three measurements",
        )
        p.add_argument("--warmup-runs", type=int)
        p.add_argument("--repeats", type=int)
    p = sub.add_parser("diagnose")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--case", required=True)
    p = sub.add_parser("report")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--leaderboard", type=Path)
    args = parser.parse_args()
    try:
        if args.command in {"plan", "e2e"}:
            cfg = load(args.config)
            if args.solutions:
                if set(args.solutions) - set(cfg["solutions"]):
                    raise ValueError(
                        "Selected solution is not in the run configuration"
                    )
                cfg["solutions"] = args.solutions
            if args.smoke:
                cfg["measurement"] = {"warmup_runs": 2, "repeats": 3}
            for key in ["warmup_runs", "repeats"]:
                if getattr(args, key) is not None:
                    minimum = 0 if key == "warmup_runs" else 1
                    if getattr(args, key) < minimum:
                        raise ValueError(f"{key} must be >= {minimum}")
                    cfg["measurement"][key] = getattr(args, key)
            selected = select_cases(cfg, args.case, args.smoke)
            if args.command == "plan":
                for case in selected:
                    print(f"{case['id']}  solutions={','.join(cfg['solutions'])}")
                print(
                    f"{len(selected)} cases; {len(selected) * len(cfg['solutions'])} jobs; {cfg['measurement']}"
                )
            else:
                _, manifest = e2e(cfg, selected)
                return int(any(j["status"] != "success" for j in manifest["jobs"]))
        elif args.command == "diagnose":
            from diagnose import run

            _, records = run(args.run, args.case)
            return int(
                any(r["summary"]["status"] in {"failed", "timed_out"} for r in records)
            )
        elif args.command == "report":
            from report import leaderboard, render

            if args.leaderboard:
                leaderboard(args.run, args.leaderboard)
            else:
                render(args.run)
            print(args.run / "report.md")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
