"""ROLE: EVALUATION ENGINEER  |  STAGE 01 - MACHINE LEARNING
Owner: Role 4 - Evaluation Engineer
Full instructions: docs/roles/ROLE_4_EVALUATION_ENGINEER.md  (find the section for Day 1)

WHAT THIS SCRIPT MUST DO
    Evaluate on the TEST split at the cost-optimal threshold chosen on VAL; compare against baselines; calibration; slices; leakage tests; threshold decision memo.

INPUT : Stage01_ML/predictions/ml_predictions.csv
OUTPUT: Stage01_ML/reports/eval_*.json|md|png + claims/CLAIMS_LEDGER.md rows

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
    print("NOT_IMPLEMENTED: 04_evaluation_engineer.py (Stage 01). See docs/roles/ROLE_4_EVALUATION_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
