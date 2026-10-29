# AGENTS.md - rules for every AI coding assistant in this repository

This file is read by Claude Code, Codex, Antigravity, Gemini CLI, Cursor, Copilot, Windsurf and any other
assistant. A human teammate is driving; you (the assistant) work on their device. **Obey this file before
doing anything else.** If a rule here conflicts with what the human just asked, tell them about the conflict
instead of silently breaking the rule.

## 0. Project in three lines

- **CASCADE**: AI-powered prediction of global supply-chain logistics bottlenecks (problem statement 07).
  We compete against other teams on the same problem; our edge is **evidence over demos** (read `docs/02_USP_CASCADE.md`).
- Five stages on five days: **Day 1 ML, Day 2 DL, Day 3 NLP, Day 4 SLM, Day 5 GenAI.** (Stage 06 Agentic AI is out of scope.)
- Four teammates, four roles, four devices. They integrate **only** through the contracts in `contracts/`.

## 1. First action in every session: find out who you are working for

1. If the human has not said which role they have, **ask**: "Which role are you: 1 ML Engineer, 2 EDA Engineer, 3 Data & Integration Engineer, or 4 Evaluation Engineer?"
2. Read your role file **completely** before writing code:
   - Role 1 -> `docs/roles/ROLE_1_ML_ENGINEER.md`
   - Role 2 -> `docs/roles/ROLE_2_EDA_ENGINEER.md`
   - Role 3 -> `docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md`
   - Role 4 -> `docs/roles/ROLE_4_EVALUATION_ENGINEER.md`
3. Then read `docs/04_GITHUB_WORKFLOW.md` and `docs/08_COMMENTING_STANDARD.md`.
4. Run `git pull --rebase` and `python scripts/board.py` to see what the other roles have handed over.
5. Reference material (the example project and the problem-statement PDF) is in `_reference/` on this device. It is read-only inspiration for **structure**, never copy its contents or its data into the repo.

## 2. The hard rules (no exceptions)

### R1. No AI attribution in git. Ever.
- Never add `Co-authored-by: Claude ...`, `Co-authored-by: Copilot/Gemini/Codex/Cursor/...`, `Generated with Claude Code`, a robot emoji line, or any similar trailer to a **commit message, PR title, PR description or file**.
- Commits are authored by the human teammate only. If your tool adds attribution by default, turn it off for this repo (`.claude/settings.json` already does it for Claude Code) and remove it from the message before committing.
- The git hooks in `.githooks/` strip such lines and CI fails if one gets through. One-time setup per clone: `git config core.hooksPath .githooks`.

### R2. Stay in your lane.
- Each role owns specific files (see `contracts/ownership.json` and section 4 of `docs/01_ARCHITECTURE.md`). **Never edit a file owned by another role.** If you need a change there, write the request in your HANDOFF file under "Open questions" or message the owner.
- Commit subjects use the owner type: `data`, `eda`, `ml`, `eval`, `integ`, plus `docs`, `chore`, `ci`. Format: `<type>(<stage>): <message>` e.g. `ml(stage01): train xgboost baseline`.

### R3. Contracts are law.
- The files in `contracts/` (node registry, schemas, label/cost/feature policies) are frozen. Do not edit them. Propose changes to Role 3 in writing.
- Every file you hand to another role must pass `python scripts/validate_contract.py <schema> <file>` first. Paste the result into your HANDOFF file.
- Use `common.paths` for every folder; never write an absolute path. Use `common.seed.set_global_seed()` (seed 42) in every script.

### R4. Comment every step in plain English.
- Every file starts with a docstring: what it does, what it reads, what it writes.
- Every function gets a one-line docstring. Every logical step inside code gets a short comment that says **what is happening and why**, written for a first-year student. Full standard with examples: `docs/08_COMMENTING_STANDARD.md`.
- Do not delete comments written by teammates. Do not leave commented-out code.

### R5. Honest numbers only.
- Every metric, table or chart in a report must be produced by code in this repo and saved to a file. Never type a metric by hand. Never invent results.
- Never train on the test split. Never choose a threshold on the test split. Never use a forbidden feature (`contracts/feature_policy.json`).
- If something is synthetic, the data says so (`data_origin`, `is_synthetic`). If a result looks too good (>0.97 on a hard task), assume leakage and investigate before celebrating.
- Never silently drop rows or clip values. Flag and log what you changed (see `data_quality_report.json` convention in the Data Engineer role file).

### R6. Safety principle of the product.
- The system is **decision support**. No code path may execute a reroute, booking or shipment change. Every prediction response carries `requires_operator_review: true`.

### R7. Secrets and size.
- Never commit `.env`, API keys, tokens, or anything from `_reference/`. Copy `.env.example` to `.env` locally.
- Files over 50 MB are blocked by the pre-commit hook. Shrink models (LoRA adapter only, quantise) or ask Role 3.

### R8. Git hygiene.
- Work on your own branch: `<role>/<stage>-<topic>` e.g. `ml/stage01-baselines`. Never commit directly to `main`. Never force-push. Never rewrite a branch that someone else has pulled.
- Before every push: `git pull --rebase origin main`, run `python -m pytest tests -q`, then push. Full procedure: `docs/04_GITHUB_WORKFLOW.md`.
- Small commits, one idea each, message in plain English.

### R9. Reproducibility.
- A teammate must be able to run your script from a clean clone with one command from inside the stage folder, e.g. `python 03_ml_engineer.py`, and get the same outputs.
- Scripts are the source of truth. Notebooks are allowed only as scratch work in `notebooks/<role>/` and are never inputs to other roles.

### R10. Do not over-build.
- Deliver the `[MUST]` items of your role file first, then `[SHOULD]`, then `[STRETCH]`. Report honestly what is missing in your HANDOFF file instead of hiding it.

## 3. Quick command card

```bash
git config core.hooksPath .githooks          # once per clone (enables the safety hooks)
git pull --rebase origin main                # start of every work session
python scripts/board.py                      # who has handed over what
python scripts/validate_contract.py --list   # list contract names
python scripts/validate_contract.py telemetry Stage01_ML/data/prepared_dataset.csv
python scripts/check_ownership.py --range origin/main..HEAD
python -m pytest tests -q
```

## 4. When unsure

Stop and ask the human, quoting the rule you are unsure about. A short question costs seconds; a wrong
push to a shared repo costs the whole team an afternoon, and a debate day cannot be repeated.
