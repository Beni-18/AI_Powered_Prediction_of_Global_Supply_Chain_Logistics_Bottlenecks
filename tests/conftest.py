"""pytest setup: lets tests import `common` and the helper scripts from any folder."""

import os
import sys

# PyTorch (Stage 2 tests) and XGBoost/LightGBM (Stage 1 tests) each bring their own OpenMP library; on macOS they
# deadlock or crash in one test process unless OpenMP is limited to one thread. Must be set before they load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
