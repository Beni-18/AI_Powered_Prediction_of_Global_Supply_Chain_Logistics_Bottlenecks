# Copilot instructions

The complete rules for this repository are in `AGENTS.md`. Read it first, then the user's role file in `docs/roles/`.

Most important rules:

1. Never add `Co-authored-by: Copilot` (or any AI tool) or "Generated with ..." text to commit messages or PR descriptions. Commits are authored by the human teammate only.
2. Only edit files owned by the user's role (`contracts/ownership.json`). Commit subjects: `<type>(<stage>): <message>`, type one of data, eda, ml, eval, integ, docs, chore, ci.
3. Write a plain-English comment for every logical step, and a docstring for every file and function (`docs/08_COMMENTING_STANDARD.md`).
4. Never edit `contracts/`. Validate handed-over files with `python scripts/validate_contract.py <schema> <file>`.
5. All reported numbers must come from code. Never use the test split for training or tuning.
