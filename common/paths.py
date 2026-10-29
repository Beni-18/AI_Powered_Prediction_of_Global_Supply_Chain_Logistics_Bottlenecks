"""Single source of truth for folder locations.

Why this file exists: the four teammates work on four different computers, so no script may
contain an absolute path like C:/Users/... . Every script asks this module where things live.
"""

from pathlib import Path

# The repository root is the folder that contains this "common" package.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared contract files (schemas, node registry, policies, tiny example files).
CONTRACTS_DIR = REPO_ROOT / "contracts"
SCHEMAS_DIR = CONTRACTS_DIR / "schemas"
FIXTURES_DIR = CONTRACTS_DIR / "fixtures"
NODES_CSV = CONTRACTS_DIR / "network_nodes.csv"
EDGES_CSV = CONTRACTS_DIR / "network_edges.csv"

# Stage number -> folder name. Same names as the reference project so the structure feels familiar.
STAGES = {
    1: "Stage01_ML",
    2: "Stage02_DL",
    3: "Stage03_NLP",
    4: "Stage04_SLM",
    5: "Stage05_GenAI",
}

# The sub-folders that exist inside every stage folder, and who writes into each one.
STAGE_SUBDIRS = {
    "data": "Role 3 (Data Engineer) writes prepared datasets here",
    "artifacts": "Role 1 (Model Engineer) writes trained models here",
    "predictions": "Role 1 writes model outputs that Role 4 evaluates",
    "reports": "Roles 2, 3 and 4 write reports here, each with their own file prefix (eda_, integ_, eval_)",
    "HANDOFF": "Every role writes one status file here for the next role",
}


def stage_dir(stage: int) -> Path:
    """Return the folder of a stage, e.g. stage_dir(1) -> <repo>/Stage01_ML."""
    return REPO_ROOT / STAGES[stage]


def stage_path(stage: int, sub: str, filename: str = "") -> Path:
    """Return a path inside a stage sub-folder, e.g. stage_path(1, 'data', 'prepared_dataset.csv')."""
    if sub not in STAGE_SUBDIRS:
        raise ValueError(f"Unknown sub-folder '{sub}'. Use one of {sorted(STAGE_SUBDIRS)}")
    folder = stage_dir(stage) / sub
    return folder / filename if filename else folder
