"""ROLE: EDA ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 2 - EDA Engineer
Full instructions: docs/roles/ROLE_2_EDA_ENGINEER.md  (find the section for Day 2)

WHAT THIS SCRIPT MUST DO
    Study autocorrelation, seasonality, node-to-node lead/lag, the 5 ambiguous delay graphs; after the model exists, run attribution and the noise-channel sanity test.

INPUT : Stage02_DL/data/hourly_node_telemetry.csv (+ model for attribution)
OUTPUT: Stage02_DL/reports/eda_*.png|csv|md

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
    print("NOT_IMPLEMENTED: 02_eda_engineer.py (Stage 02). See docs/roles/ROLE_2_EDA_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
