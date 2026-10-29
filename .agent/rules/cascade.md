# CASCADE team rules (Antigravity workspace rule: always on)

The complete rules for this repository are in `AGENTS.md` at the repository root. Read it first, then the user's role file in `docs/roles/`.

1. Never add AI attribution (`Co-authored-by: Gemini/Claude/...`, "Generated with ...") to commits or PR text.
2. Edit only files your role owns (`contracts/ownership.json`). Commit subject: `<type>(<stage>): <message>`.
3. Plain-English comment on every logical step; docstring on every file and function (`docs/08_COMMENTING_STANDARD.md`).
4. `contracts/` is frozen. Validate handed-over files with `python scripts/validate_contract.py <schema> <file>`.
5. Every number in a report comes from code. Never train or tune on the test split.
