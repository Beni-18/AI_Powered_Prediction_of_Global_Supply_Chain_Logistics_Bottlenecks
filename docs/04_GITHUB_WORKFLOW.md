# 04 - GitHub workflow (read this before your first push)

Repository: `https://github.com/Beni-18/AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks`
Everyone is a collaborator with write access. **Nobody commits to `main` directly.** This document is binding
for humans and for every AI assistant (Claude Code, Antigravity, Codex, Cursor, Copilot, Gemini ...).

## 1. Golden rules (memorise these five)

1. **No AI attribution, ever.** No `Co-authored-by: Claude` (or Copilot, Gemini, Codex, Cursor ...), no "Generated with Claude Code", no robot-emoji line - in commit messages, PR titles, PR descriptions. Commits are authored by the human teammate only.
2. **Own branch, small commits, pull before push.** Branch `<role>/<stage>-<topic>`; `git pull --rebase origin main` before every push.
3. **Only touch your own files.** Ownership map: `contracts/ownership.json`. Lane violations fail CI.
4. **Never force-push, never rewrite shared history.**
5. **Hand-off = commit + push + update your HANDOFF file.** Others only trust what is on GitHub.

## 2. One-time setup on each device (15 minutes, do it on Day 0)

```bash
# 1. clone
git clone https://github.com/Beni-18/AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks.git
cd AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks

# 2. tell git who YOU are (your own name and your own GitHub e-mail; never anything about an AI tool)
git config user.name  "Your Real Name"
git config user.email "you@example.com"

# 3. switch on the safety hooks (strip AI attribution, block secrets/huge files, enforce commit format)
git config core.hooksPath .githooks
# macOS/Linux only, if hooks do not run:  chmod +x .githooks/*

# 4. python environment
python -m venv .venv
#   Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 5. proof that everything works
python -m pytest tests -q
python scripts/validate_contract.py --self-test
python scripts/board.py
```

Then copy the reference project and the problem-statement PDF into `_reference/` (see `_reference/README.md`) and copy `.env.example` to `.env` if you need keys.

**Authentication:** when git asks for a password, use a GitHub *personal access token* (Settings -> Developer settings -> Personal access tokens, scope `repo`), or sign in with GitHub Desktop / `gh auth login`. Never put the token in a file in the repo.

## 3. Branches

| Purpose | Name pattern | Example |
|---|---|---|
| Daily stage work | `<role>/<stage>-<topic>` | `ml/stage01-baselines`, `eda/stage02-lead-lag`, `data/stage03-corpus`, `eval/stage04-faithfulness`, `integ/stage05-injection-panel` |
| Fix after a hand-off | `<role>/fix-<topic>` | `data/fix-split-leak` |

`<role>` is exactly the commit type of your role: `ml`, `eda`, `data`, `eval`, `integ`.
Role 3 uses `data/...` for data-hat work and `integ/...` for integration-hat work.
Branch from the latest `main`, keep it alive for one stage-day at most, delete it after it is merged.

## 4. Commit messages

Format (the hook rejects anything else):

```
<type>(<stage>): <what you did, in plain English, present tense>

Optional body: why, what you decided, what the next role must know.
Cross-role: <only when you must touch another role's file: who agreed and why>
```

| type | who | covers |
|---|---|---|
| `data` | Role 3 | `01_data_engineer.py`, `Stage*/data/**`, `data/**`, `contracts/**`, `common/dataio/**`, fetch scripts |
| `eda` | Role 2 | `02_eda_engineer.py`, `reports/eda_*`, `common/edakit/**`, `scenarios/**` |
| `ml` | Role 1 | `03_*_engineer.py`, `artifacts/**`, `predictions/**`, `common/modelkit/**`, stage `requirements.txt` |
| `eval` | Role 4 | `04_evaluation_engineer.py`, `reports/eval_*`, `common/evalkit/**`, `claims/**` |
| `integ` | Role 3 | `05_integration_engineer.py`, `integration/**`, `app.py`, `pages/**`, `run_pipeline.py`, root `requirements*.txt` |
| `docs` | anyone | markdown only (`docs/**`, READMEs) |
| `chore` / `ci` | Role 3 (rare) | scripts, ignore files / `.github`, hooks |

Good: `ml(stage01): add class-weighted xgboost and write ml_predictions.csv`
Bad: `Update README.md`, `fix`, `final version`, `ml: stuff (generated with Claude)`.

**Automatic protection:** `.githooks/commit-msg` deletes AI attribution lines and rejects bad subjects.
`.githooks/pre-push` refuses to push if a commit still contains attribution. CI repeats both checks.

### Telling your AI assistant (paste this into the chat if it ever adds attribution)

> Never add Co-authored-by, "Generated with", or any AI/tool attribution to commit messages or PR text in this repo.
> Use `git commit -m "<type>(<stage>): <message>"` exactly, with nothing appended. If your tool has a setting for
> commit attribution, turn it off. Claude Code: `.claude/settings.json` already sets `attribution.commit` and `attribution.pr` to empty.
> Cursor/Copilot/Gemini/Antigravity/Codex: do not add trailers when writing commit messages or PR descriptions.

## 5. The daily rhythm

```bash
# --- start of the work session ---
git checkout main && git pull --rebase origin main        # latest shared state
python scripts/board.py                                    # what has been handed over?
git checkout -b ml/stage01-baselines                       # your branch for today

# --- while working: commit small and often ---
git add Stage01_ML/03_ml_engineer.py Stage01_ML/predictions/ml_predictions.csv   # add files by name, not "git add ."
git commit -m "ml(stage01): add rule and logistic baselines"

# --- hand-off moment ---
python scripts/validate_contract.py ml_predictions Stage01_ML/predictions/ml_predictions.csv
# edit Stage01_ML/HANDOFF/03_model_engineer.md : status: READY, files, decisions, paste the validator output
git add Stage01_ML/HANDOFF/03_model_engineer.md
git commit -m "ml(stage01): hand off predictions to evaluation"

git fetch origin && git rebase origin/main                 # replay your commits on the newest main
python -m pytest tests -q
git push -u origin ml/stage01-baselines                    # first push of the branch
# open the Pull Request on GitHub (title in commit format) -> CI must be green -> merge
```

**`git add .` is discouraged.** Add by name, or `git add Stage01_ML/` if the whole folder is yours. Check
`git status` first; it must list only your files.

## 6. Pull requests and merging

- One PR per role per stage-day. Title in commit format: `ml(stage01): xgboost baseline + calibration`.
- PR description: fill the template (what, role/stage, hand-off checklist). **No AI attribution text.**
- CI (`guard`) must be green: no attribution, lanes respected, contracts valid, tests pass.
- Merge method: **"Create a merge commit"** (not squash, not rebase) so each role's commit messages survive.
- **Who merges:** the author merges their own PR once CI is green and the hand-off file says READY. Role 3 may merge on behalf of anyone during a time crunch. A review by one other teammate is welcome but not required for files in your own lane; it **is** required for anything in `contracts/` (Role 3 proposes, one other role approves).
- After merge: tell the next role in the group chat ("Day 1 data is on main"), delete the branch.

## 7. How hand-offs work (the relay)

Inside a stage the files travel `01 -> 02 -> 03 -> 04 -> 05`, but nobody waits idle: every role starts
immediately against the **contract examples** in `contracts/fixtures/` and swaps in the real file when the
previous role's PR is merged. Full protocol: `docs/09_HANDOFF_PROTOCOL.md`.

## 8. What must NEVER be committed

| Never | Why | Instead |
|---|---|---|
| `.env`, API keys, tokens, `*.pem` | Secrets leak forever in git history | `.env` stays local; `.env.example` is the template |
| `_reference/` content | Licence + size + confusion with our own work | Local only (git-ignored) |
| Files > 50 MB (hook blocks) | GitHub rejects > 100 MB | Quantise, save LoRA adapter only, sample, or ask Role 3 |
| `data/raw/real/**` | Re-creatable, licence-bound | Commit the fetch script and `docs/06_DATA_SOURCES.md` entry |
| Training checkpoints, `wandb/`, `mlruns/` | Huge, useless to others | Commit only the final artefact + `model_card.json` |
| Absolute paths, other people's usernames in code | Breaks on other devices | `common.paths` |

Small CSV/JSON/PNG results and final model artefacts under 50 MB **are** committed on purpose: that is how
the next role (on another device) receives them.

## 9. Conflict handling

Conflicts should be rare because files are owned. If one happens anyway:

```bash
git fetch origin
git rebase origin/main              # git stops at the conflicting commit
git status                          # lists the files in conflict
# open each file, look for <<<<<<< ======= >>>>>>>, keep the correct content, delete the markers
git add <file>
git rebase --continue
```

- If the conflicting file is **not yours**, you probably edited someone else's file: abort (`git rebase --abort`), undo your edit to it (`git checkout origin/main -- <file>`), and ask the owner.
- If it is a data/model file (CSV, `.pt`, `.pkl`), never hand-merge: take the owner's latest (`git checkout origin/main -- <file>`) and regenerate yours.
- Never "resolve" by force-pushing.

## 10. Recovery recipes

**An AI co-author line slipped into a commit that is NOT pushed yet**

```bash
git rebase origin/main --exec "python scripts/strip_ai_trailers.py"
```

**It was already pushed to YOUR branch** (nobody else uses your branch): run the line above, then
`git push --force-with-lease origin <your-branch>`. This is the only allowed use of force, and only on your own branch.
**It reached `main`:** tell Role 3 immediately; do not try to fix it yourself.

**Committed to `main` by mistake (not pushed):**
```bash
git branch ml/stage01-rescue          # keep the work on a new branch
git reset --hard origin/main          # put local main back
git checkout ml/stage01-rescue
```

**Pushed the wrong file / a secret:** tell Role 3 at once. A leaked key must be revoked by its owner first; deleting the file does not remove it from history.

**"I broke my branch":** `git reflog` shows where you were; `git reset --hard <sha>` goes back.

**Behind origin:** `git pull --rebase origin main`.

## 11. Repository owner checklist (Role 3 / repo admin does this once, on Day 0)

On GitHub -> Settings:
- [ ] Collaborators: add all four teammates with **Write** access.
- [ ] Branches -> add rule for `main`: *Require a pull request before merging* (0 required approvals), *Require status checks to pass* -> select `guard`, *Do not allow force pushes*, *Do not allow deletions*.
- [ ] General -> Pull Requests: enable **Allow merge commits**; disable **Allow squash merging** and **Allow rebase merging**; enable *Automatically delete head branches*.
- [ ] Actions -> General: allow workflows to run (the `guard` workflow lives in `.github/workflows/guard.yml`).
- [ ] Optional: after everyone's GitHub username is known, copy `docs/CODEOWNERS.template` to `.github/CODEOWNERS` and fill in the usernames.

## 12. IDE-specific notes

| IDE / assistant | Reads rules from | Attribution switch |
|---|---|---|
| Claude Code | `CLAUDE.md` (imports `AGENTS.md`) | `.claude/settings.json` already sets empty `attribution.commit` / `attribution.pr` |
| Google Antigravity / Gemini CLI | `GEMINI.md`, `AGENTS.md`, `.agent/rules/` | Do not let it append trailers; paste the block in section 4 if it does |
| Codex | `AGENTS.md` | Same |
| Cursor | `.cursor/rules/cascade.mdc`, `AGENTS.md` | Disable any "commit message attribution" option |
| GitHub Copilot | `.github/copilot-instructions.md` | Same |
| Anything else | Paste the matching prompt from `docs/07_KICKOFF_PROMPTS.md` as the first message | Same |

If your assistant does not auto-load rule files, **the first message of every session must be your role's kickoff prompt** from `docs/07_KICKOFF_PROMPTS.md`.
