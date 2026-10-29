"""ROLE: INTEGRATION ENGINEER  |  STAGE 05 - GENERATIVE AI
Owner: Role 3 - Data & Integration Engineer
Full instructions: docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md  (find the section for Day 5)

WHAT THIS SCRIPT MUST DO
    Build the scenario-injection control panel, wildcard live-run view, final fused control tower and run_pipeline.py.

INPUT : Everything above
OUTPUT: app.py + pages/* + Stage05_GenAI/integration/*

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
    print("NOT_IMPLEMENTED: 05_integration_engineer.py (Stage 05). See docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
