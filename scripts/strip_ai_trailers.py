"""Removes AI attribution lines from the message of the commit currently checked out (HEAD).

Why this file exists: if an IDE added "Co-authored-by: Claude ..." to local commits that were not pushed yet,
this rewrites them cleanly. Use it with an interactive-free rebase:

    git rebase origin/main --exec "python scripts/strip_ai_trailers.py"

It amends HEAD only when something had to be removed, so it is safe to run on already-clean commits.
"""

import re
import subprocess
import sys

# Same patterns as .githooks/commit-msg so the two never disagree.
CO_AUTHOR = re.compile(
    r"^\s*co-authored-by:.*(claude|anthropic|copilot|chatgpt|openai|codex|gemini|cursor|windsurf|cline|aider|devin|\[bot\]|noreply@anthropic)",
    re.IGNORECASE,
)
GENERATED = re.compile(
    r"(generated|created|written|made|authored)\s+(with|by).*(claude|copilot|chatgpt|codex|gemini|cursor|windsurf|aider)",
    re.IGNORECASE,
)


def clean(message: str) -> str:
    """Return the message without AI attribution lines and without trailing blank lines."""
    kept = [line for line in message.splitlines() if not CO_AUTHOR.search(line) and not GENERATED.search(line)]
    return "\n".join(kept).rstrip() + "\n"


def main() -> int:
    current = subprocess.run(["git", "log", "-1", "--format=%B"], capture_output=True, text=True, check=True).stdout
    cleaned = clean(current)
    if cleaned.strip() == current.strip():
        return 0  # nothing to remove
    # --no-verify: the commit-msg hook would only re-check what we just cleaned.
    subprocess.run(["git", "commit", "--amend", "--no-verify", "-F", "-"], input=cleaned, text=True, check=True)
    print("removed AI attribution from", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
