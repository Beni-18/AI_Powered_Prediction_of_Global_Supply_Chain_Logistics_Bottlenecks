"""ROLE: ML ENGINEER  |  STAGE 01 - MACHINE LEARNING
Owner: Role 1 - ML Engineer (model role)
Full instructions: docs/roles/ROLE_1_ML_ENGINEER.md  (find the section for Day 1)

WHAT THIS SCRIPT MUST DO
    Train the baselines first (rule, logistic), then Random Forest, XGBoost, Isolation Forest, One-Class SVM; calibrate; write predictions in the ml_predictions contract.

INPUT : Stage01_ML/data/prepared_dataset.csv + contracts/feature_policy.json
OUTPUT: Stage01_ML/predictions/ml_predictions.csv + Stage01_ML/artifacts/* + model_card.json

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
    print("NOT_IMPLEMENTED: 03_ml_engineer.py (Stage 01). See docs/roles/ROLE_1_ML_ENGINEER.md.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
