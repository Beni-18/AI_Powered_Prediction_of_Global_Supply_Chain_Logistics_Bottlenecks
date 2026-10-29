"""Prints the team status board by reading every Stage*/HANDOFF/*.md file.

Why this file exists: instead of editing one shared status file (which causes merge conflicts),
each role edits only its own HANDOFF file. This script reads them all and shows one table.

Usage:
    git pull --rebase
    python scripts/board.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import REPO_ROOT, STAGES  # noqa: E402

# Column order = order of the relay inside one stage.
COLUMNS = [
    ("01", "data_engineer", "Data"),
    ("02", "eda_engineer", "EDA"),
    ("03", "model_engineer", "Model"),
    ("04", "evaluation_engineer", "Eval"),
    ("05", "integration_engineer", "Integ"),
]
STATUS_RE = re.compile(r"^status:\s*([A-Z_]+)", re.MULTILINE)
SHORT = {"NOT_STARTED": "-", "IN_PROGRESS": "WIP", "READY": "READY", "BLOCKED": "BLOCKED"}


def read_status(path: Path) -> str:
    """Return the status word of one HANDOFF file, or '?' when the file or line is missing."""
    if not path.exists():
        return "?"
    match = STATUS_RE.search(path.read_text(encoding="utf-8"))
    return SHORT.get(match.group(1), match.group(1)) if match else "?"


def main() -> int:
    header = ["Stage"] + [c[2] for c in COLUMNS]
    rows = []
    for number, folder in STAGES.items():
        cells = [folder]
        for prefix, name, _ in COLUMNS:
            cells.append(read_status(REPO_ROOT / folder / "HANDOFF" / f"{prefix}_{name}.md"))
        rows.append(cells)

    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))  # noqa: E731
    print(line(header))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))
    print("\nLegend: - not started | WIP in progress | READY handed over | BLOCKED needs help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
