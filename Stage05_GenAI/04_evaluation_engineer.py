"""ROLE: EVALUATION ENGINEER  |  STAGE 05 - GENERATIVE AI
Owner: Role 4 - Evaluation Engineer
Full instructions: docs/roles/ROLE_4_EVALUATION_ENGINEER.md  (find the section for Day 5)

WHAT THIS SCRIPT MUST DO
    Realism audit (ranges, conservation, fidelity, TSTR, discriminator test) and the 20-scenario stress test through Stages 1-4.

INPUT : Stage05_GenAI/predictions/scenario_telemetry.csv + all earlier artifacts
OUTPUT: Stage05_GenAI/reports/eval_stress_results.csv (contract: stress_results) + eval_*.md|png

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
    print("NOT_IMPLEMENTED: 04_evaluation_engineer.py (Stage 05). See docs/roles/ROLE_4_EVALUATION_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
