# GEMINI.md  (read by Gemini CLI and Google Antigravity)

The complete rules for this repository are in **`AGENTS.md`** - open it and follow it before doing anything else.
Then read your role file in `docs/roles/` (ask the user which role they have if they have not said).

The five rules that matter most (the full list is in AGENTS.md):

1. **No AI attribution in git**: never add `Co-authored-by: Gemini/Claude/Copilot/...` or "Generated with ..." lines to commits or PR text. Commits are authored by the human teammate only.
2. **Stay in your lane**: edit only files your role owns (`contracts/ownership.json`). Commit subject format: `<type>(<stage>): <message>` with type one of data, eda, ml, eval, integ, docs, chore, ci.
3. **Plain-English comments** on every logical step and a docstring on every file and function (`docs/08_COMMENTING_STANDARD.md`).
4. **Contracts are frozen** (`contracts/`). Validate every handed-over file with `python scripts/validate_contract.py <schema> <file>`.
5. **Honest numbers only**: all metrics come from code; never train or tune on the test split; never use forbidden features.
