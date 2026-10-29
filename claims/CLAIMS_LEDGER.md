# Claims ledger

**Rule:** we only say a number aloud in a debate if it has a row here with status `VERIFIED`.
Owner: Role 4 (Evaluation Engineer). Rows are written by code (`common/evalkit` `ledger.upsert(...)`), not by hand.
`Reproduce command` must work from a clean clone. `Verified on` = date a clean-clone run reproduced the value.

Status values: `PENDING` (computed, not yet re-run clean) · `VERIFIED` · `RETRACTED` (kept for honesty, never quoted).

<!-- LEDGER:BEGIN -->
| ID | Day | Claim wording | Value [95% CI] | Source file | Reproduce command | Verified on | Status |
|---|---|---|---|---|---|---|---|
<!-- LEDGER:END -->
