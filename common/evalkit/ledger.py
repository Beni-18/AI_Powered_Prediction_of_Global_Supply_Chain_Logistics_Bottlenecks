"""The claims ledger: every number we say aloud in a debate has one row here, written by code.

Why this file exists: hand-edited tables drift away from the real results and cause merge conflicts.
`upsert` rewrites exactly one row between the marker lines, so re-running a script refreshes its claim.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from common.paths import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "claims" / "CLAIMS_LEDGER.md"
BEGIN, END = "<!-- LEDGER:BEGIN -->", "<!-- LEDGER:END -->"
HEADER = [
    "| ID | Day | Claim wording | Value [95% CI] | Source file | Reproduce command | Verified on | Status |",
    "|---|---|---|---|---|---|---|---|",
]
STATUSES = {"PENDING", "VERIFIED", "RETRACTED"}


def _cell(text) -> str:
    """Make a value safe inside a markdown table cell."""
    return str(text).replace("|", "/").replace("\n", " ").strip()


def _split_ledger(text: str) -> tuple:
    """Return (text before BEGIN, the table lines, text after END)."""
    head, rest = text.split(BEGIN, 1)
    table, tail = rest.split(END, 1)
    return head + BEGIN, [ln for ln in table.strip().splitlines() if ln.strip()], END + tail


def upsert(claim_id: str, day: int, claim: str, value: str, source_file: str, reproduce: str,
           status: str = "PENDING", verified_on: str = "", path: Path | None = None) -> None:
    """Insert or replace the row with this ID. Rows stay sorted by ID so diffs are small."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    path = path or LEDGER_PATH
    head, lines, tail = _split_ledger(path.read_text(encoding="utf-8"))

    # Keep only data rows (drop the header and separator), then drop any old row with this ID.
    rows = [ln for ln in lines if ln.startswith("|") and not ln.startswith(("| ID", "|---"))]
    rows = [ln for ln in rows if ln.split("|")[1].strip() != claim_id]

    if status == "VERIFIED" and not verified_on:
        verified_on = date.today().isoformat()
    new_row = "| " + " | ".join(_cell(x) for x in
                                 [claim_id, day, claim, value, source_file, f"`{reproduce}`", verified_on, status]) + " |"
    rows.append(new_row)
    rows.sort(key=lambda ln: ln.split("|")[1].strip())

    path.write_text(head + "\n" + "\n".join(HEADER + rows) + "\n" + tail, encoding="utf-8")


def read_rows(path: Path | None = None) -> list:
    """Return the ledger as a list of dicts (used by tests and by the pre-debate check)."""
    path = path or LEDGER_PATH
    _, lines, _ = _split_ledger(path.read_text(encoding="utf-8"))
    cols = ["id", "day", "claim", "value", "source", "reproduce", "verified_on", "status"]
    out = []
    for ln in lines:
        if ln.startswith("|") and not ln.startswith(("| ID", "|---")):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            out.append(dict(zip(cols, cells)))
    return out
