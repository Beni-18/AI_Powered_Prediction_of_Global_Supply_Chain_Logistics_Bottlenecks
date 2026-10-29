"""ROLE: EDA ENGINEER  |  STAGE 01 - MACHINE LEARNING
Owner: Role 2 - EDA Engineer
Full instructions: docs/roles/ROLE_2_EDA_ENGINEER.md  (find the section for Day 1)

WHAT THIS SCRIPT MUST DO
    Find the noise floor, seasonality and reporting drift; run the leakage audit; write feature recommendations for the ML Engineer.

INPUT : Stage01_ML/data/prepared_dataset.csv
OUTPUT: Stage01_ML/reports/eda_*.csv|png|json + eda_findings.md

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
    print("NOT_IMPLEMENTED: 02_eda_engineer.py (Stage 01). See docs/roles/ROLE_2_EDA_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
