# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- The full rules are in `AGENTS.md` (imported above). Read your role file in `docs/roles/` before coding.
- **Never add a `Co-Authored-By: Claude ...` trailer or "Generated with Claude Code" line** to commits or PR descriptions. `.claude/settings.json` turns the default attribution off; do not override it. If you wrote a commit message with such a line, rewrite the message before committing.
- Use the plain-English comment standard (`docs/08_COMMENTING_STANDARD.md`) in every file you write.
- Stay inside the files your role owns (`contracts/ownership.json`). Ask the human before touching anything else.
