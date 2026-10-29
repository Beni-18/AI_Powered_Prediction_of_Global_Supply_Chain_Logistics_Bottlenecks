"""Runs the role scripts 01-04 of every stage in order and prints a summary table.

Why this file exists: one command proves the whole relay still works on a clean checkout.
Script 05 (the API / app) is NOT run here because it starts a server; start it by hand.

Usage:
    python run_pipeline.py                  run every stage
    python run_pipeline.py --stage 1        run only Stage 1
    python run_pipeline.py --from-stage 2   run Stage 2 and later
    python run_pipeline.py --skip 4 5       skip those stages

Exit codes of a role script:  0 = done,  3 = NOT_IMPLEMENTED (skipped),  anything else = failure (pipeline stops).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common.paths import REPO_ROOT, STAGES

NOT_IMPLEMENTED = 3
ROLE_PREFIXES = ["01", "02", "03", "04"]  # 05 starts a server, so it is excluded


def scripts_for(stage_folder: Path) -> list:
    """Find the 01..04 scripts of one stage, in numeric order."""
    found = []
    for prefix in ROLE_PREFIXES:
        found.extend(sorted(stage_folder.glob(f"{prefix}_*.py")))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=STAGES.keys(), help="run only this stage")
    parser.add_argument("--from-stage", type=int, choices=STAGES.keys(), help="run this stage and every later one")
    parser.add_argument("--skip", type=int, nargs="*", default=[], help="stage numbers to skip")
    args = parser.parse_args()

    # Decide which stages to run.
    wanted = list(STAGES)
    if args.stage:
        wanted = [args.stage]
    elif args.from_stage:
        wanted = [s for s in wanted if s >= args.from_stage]
    wanted = [s for s in wanted if s not in args.skip]

    results = []  # (stage folder, script name, outcome)
    for stage in wanted:
        folder = REPO_ROOT / STAGES[stage]
        for script in scripts_for(folder):
            print(f"\n>>> {folder.name}/{script.name}", flush=True)
            # Run from inside the stage folder, like a teammate would on their own laptop.
            code = subprocess.run([sys.executable, str(script)], cwd=folder).returncode
            if code == 0:
                results.append((folder.name, script.name, "DONE"))
            elif code == NOT_IMPLEMENTED:
                results.append((folder.name, script.name, "SKIPPED (not implemented)"))
            else:
                results.append((folder.name, script.name, f"FAILED (exit {code})"))
                break  # stop this stage; later scripts need this script's output
        if results and results[-1][2].startswith("FAILED"):
            break

    print("\n=== Pipeline summary ===")
    for folder, script, outcome in results:
        print(f"{folder:<14} {script:<28} {outcome}")
    failed = [r for r in results if r[2].startswith("FAILED")]
    if failed:
        print("\nPipeline stopped on the first failure. Fix it, then re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
