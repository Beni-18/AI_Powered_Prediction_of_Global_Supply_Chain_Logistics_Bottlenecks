# 09 - Hand-off protocol (how four devices stay in sync)

## The idea in one paragraph

Inside a stage the work flows `01 Data -> 02 EDA -> 03 Model -> 04 Evaluation -> 05 Integration`, but the
roles do **not** wait for each other. Every role starts at minute zero against the tiny example files in
`contracts/fixtures/` (they follow the exact contract), builds and tests their code, and swaps in the real
upstream file the moment it is merged to `main`. Because every hand-over file is checked by
`scripts/validate_contract.py`, "it worked on my fixture" means "it will work on the real file".

## Who waits for whom (and what to do while waiting)

| Role | First real input it needs | Until then it does |
|---|---|---|
| 3 Data (01) | Day 0 simulator output (`data/raw/sim/`) | Validates, cleans, **publishes v1 within the first hour** of each day |
| 2 EDA (02) | Raw files from `data/raw/sim/` (available since Day 0) | Starts on the raw data; re-runs on the prepared file when it lands; the model-dependent part (attribution etc.) comes last |
| 1 Model (03) | Prepared dataset v1 | Builds the training script on the fixture, trains the baselines on v1, iterates on v2 |
| 4 Eval (04) | `predictions/*.csv` in the contract format | Builds metrics code on the fixture; the first run on baseline predictions is the dress rehearsal |
| 3 Integ (05) | Model artefact (or just its interface) | Builds API/UI against a **stub predictor** that returns fixture-shaped answers, swaps in the real model |

## The HANDOFF file

Each role has one file per stage: `Stage0N/HANDOFF/0K_<role>.md` (templates exist). Only its owner edits it.

Required fields (the board script reads `status:`):

```
status: NOT_STARTED | IN_PROGRESS | READY | BLOCKED
updated_utc: 2026-11-02T10:15:00Z
git_commit: <short sha of the commit that contains the produced files>
data_version: v1            # bump to v2, v3 when you republish the same file with changes
```

Sections: files produced (path | contract | rows | notes), contract check output, decisions and assumptions,
things the next role must know, open questions for other roles.

### Status meanings

- `IN_PROGRESS`: working; a v0/v1 may already be on a branch (not main).
- `READY`: pushed **and merged to main**, contract check passed and pasted. Downstream may rely on it.
- `BLOCKED`: you cannot continue; write exactly what you need and from whom.

### Republishing (v2, v3 ...)

If you change a hand-over file after it was READY: bump `data_version`, list **what changed and why** under
"Things the next role must know", keep the schema identical (schema changes are contract changes), and
post one line in the group chat ("Stage01 prepared_dataset v2: dropped 31 duplicate records").
Downstream roles pull and re-run; they never edit your file.

## Change requests between roles

1. Write it in **your** HANDOFF file under "Open questions" (role name + the exact ask), and send the same sentence in the group chat.
2. The owner answers in **their** HANDOFF file or by doing it. Nobody edits another role's file to "help".
3. If a quick fix by someone else is truly faster, the owner says "go ahead" in the chat; the helper commits with a `Cross-role: agreed with <owner> on <date>` line.

## Contract changes (rare, formal)

Schemas, node registry and policies are frozen from Day 0. Additive changes (a new optional column, a new
node alias) are allowed through a PR by Role 3 that another teammate approves; breaking changes need a team vote
and an entry in `docs/decisions/`. Never change the meaning of an existing column.

## Daily synchronisation points

| When | What | Who |
|---|---|---|
| Start of each day | `git pull`, `python scripts/board.py`, read the previous day's HANDOFF notes | everyone |
| 60 minutes after the guide session | **Data v1 gate**: Role 3's prepared file is merged to `main` | Role 3 |
| Before the debate | **Debate-ready gate**: first end-to-end numbers exist; Role 4 updates `claims/CLAIMS_LEDGER.md`; 15-minute rehearsal | all |
| End of day | **Day-complete gate**: all five HANDOFF files READY or honestly explained; `python run_pipeline.py --stage N` runs; tag `dayN-complete` | Role 3 tags |

## Group chat etiquette

One line per event: `[Role 3] Stage02 hourly_node_telemetry v1 on main (432,000 rows, 27 nodes, contract PASS)`.
No file attachments; the repository is the source of truth.
