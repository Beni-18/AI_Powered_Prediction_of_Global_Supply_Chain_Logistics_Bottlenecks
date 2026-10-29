"""ROLE: EVALUATION ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 4 - Evaluation Engineer
Full instructions: docs/roles/ROLE_4_EVALUATION_ENGINEER.md  (find the section for Day 2)

WHAT THIS SCRIPT MUST DO
    Skill score vs persistence, lead-time metric, rolling-origin backtest, visual prediction benchmark against Stage 1 style alerts.

INPUT : Stage02_DL/predictions/dl_forecasts.csv
OUTPUT: Stage02_DL/reports/eval_*.json|png|md + ledger rows

STATUS: NOT IMPLEMENTED YET. Replace everything below the line with the real code.
Comment standard: every logical step gets a plain-English comment (see docs/08_COMMENTING_STANDARD.md).
Exit code 3 means "not implemented"; run_pipeline.py treats it as SKIPPED, not as a failure.
"""

import sys
from pathlib import Path

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import stage_path  # noqa: E402,F401  (use this instead of hard-coded folders)
from common.seed import set_global_seed  # noqa: E402


def main() -> int:
    set_global_seed()  # same random numbers on every laptop
    print("NOT_IMPLEMENTED: 04_evaluation_engineer.py (Stage 02). See docs/roles/ROLE_4_EVALUATION_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
