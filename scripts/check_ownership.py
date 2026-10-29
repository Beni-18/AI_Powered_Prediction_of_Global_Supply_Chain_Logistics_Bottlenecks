"""Checks that commits follow the message format AND stay inside their owner's lane.

Why this file exists: the team has four people on four devices. If everyone only edits files they
own, git merges never conflict. This script makes that rule checkable by the git hook and by CI.

Commit subject format:   <type>(<stage>): <short message>
    types: data, eda, ml, eval, integ   (role work)
           docs, chore, ci              (shared work)
    example:  ml(stage01): train xgboost baseline with class weights

A commit that really must touch another role's files needs this line in its body:
    Cross-role: <who agreed, and why>

Usage:
    python scripts/check_ownership.py --range origin/main..HEAD
    python scripts/check_ownership.py --commit <sha>
    python scripts/check_ownership.py --range A..B --warn-only      (print problems, always exit 0)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OWNERSHIP_FILE = REPO_ROOT / "contracts" / "ownership.json"

# type(scope)!: message   -> scope and the "!" are optional
SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(\((?P<scope>[a-z0-9_,\- ]+)\))?!?: (?P<msg>\S.*)$")


def glob_to_regex(pattern: str) -> re.Pattern:
    """Turn a simple glob ('*' = within a folder, '**' = across folders, '?' = one character) into a regex."""
    out, i = "", 0
    while i < len(pattern):
        if pattern[i:i + 2] == "**":
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile("^" + out + "$")


def matches_any(path: str, patterns: list) -> bool:
    return any(glob_to_regex(p).match(path) for p in patterns)


def load_rules() -> dict:
    return json.loads(OWNERSHIP_FILE.read_text(encoding="utf-8"))


def allowed_patterns(rules: dict, commit_type: str) -> list | None:
    """Return the path patterns a commit type may touch, or None when the type is unknown."""
    if commit_type in rules["types"]:
        return rules["types"][commit_type]["paths"]
    if commit_type == "docs":
        return rules["docs_paths"]
    if commit_type == "chore":
        return rules["chore_paths"]
    if commit_type == "ci":
        return rules["ci_paths"]
    return None


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def commits_in_range(rev_range: str) -> list:
    out = git("rev-list", "--reverse", rev_range).split()
    return out


def check_commit(sha: str, rules: dict) -> list:
    """Return a list of human readable problems for one commit (empty list = fine)."""
    parents = git("rev-list", "--parents", "-n", "1", sha).split()[1:]
    message = git("log", "-1", "--format=%B", sha)
    subject = message.splitlines()[0] if message.strip() else ""

    # Merge commits and reverts are always fine; they carry no new authorship.
    if len(parents) > 1 or subject.startswith(("Merge ", "Revert ")):
        return []

    short = sha[:8]
    match = SUBJECT_RE.match(subject)
    if not match:
        return [f"{short}: subject '{subject}' does not match '<type>(<stage>): <message>' (types: data, eda, ml, eval, integ, docs, chore, ci)"]

    commit_type = match.group("type")
    patterns = allowed_patterns(rules, commit_type)
    if patterns is None:
        return [f"{short}: unknown type '{commit_type}' (use data, eda, ml, eval, integ, docs, chore, ci)"]

    # An explicit, written agreement lets a commit cross lanes.
    if re.search(r"^Cross-role:\s*\S+", message, flags=re.MULTILINE):
        return []

    files = [f for f in git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha).splitlines() if f]
    outside = [f for f in files if not matches_any(f, patterns)]
    if not outside:
        return []

    shown = ", ".join(outside[:6]) + (f" (+{len(outside) - 6} more)" if len(outside) > 6 else "")
    return [
        f"{short}: type '{commit_type}' may not touch: {shown}. "
        f"Use the right type for those files, split the commit, or add a 'Cross-role: <who agreed>' line."
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--range", help="git revision range, e.g. origin/main..HEAD")
    group.add_argument("--commit", help="a single commit sha")
    parser.add_argument("--warn-only", action="store_true", help="print problems but exit 0")
    args = parser.parse_args()

    rules = load_rules()
    shas = commits_in_range(args.range) if args.range else [args.commit]

    problems = []
    for sha in shas:
        problems.extend(check_commit(sha, rules))

    if problems:
        print("Ownership / commit-format problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"OK: {len(shas)} commit(s) follow the format and stay in their lane.")

    if problems and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
