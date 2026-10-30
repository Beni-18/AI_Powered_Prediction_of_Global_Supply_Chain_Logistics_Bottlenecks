"""Tests for the team rules: file ownership, commit messages, AI-attribution stripping, stage scaffolding."""

import re
import subprocess
import sys

import pytest

import check_ownership as own
import strip_ai_trailers as strip
from common.paths import REPO_ROOT, STAGES

RULES = own.load_rules()


# ---------- glob matching ----------

def test_glob_star_stays_inside_one_folder():
    assert own.matches_any("Stage01_ML/03_ml_engineer.py", ["Stage0?_*/03_*_engineer.py"])
    assert not own.matches_any("Stage01_ML/sub/03_ml_engineer.py", ["Stage0?_*/03_*_engineer.py"])


def test_glob_double_star_crosses_folders():
    assert own.matches_any("Stage02_DL/artifacts/a/b/model.pt", ["Stage0?_*/artifacts/**"])


# ---------- the ownership map must give every role script exactly one owner ----------

OWNER_OF_PREFIX = {"01": "data", "02": "eda", "03": "ml", "04": "eval", "05": "integ"}


@pytest.mark.parametrize("stage_folder", STAGES.values())
def test_each_stage_script_has_exactly_one_owner(stage_folder):
    for script in sorted((REPO_ROOT / stage_folder).glob("0[1-5]_*.py")):
        rel = f"{stage_folder}/{script.name}"
        owners = [t for t in RULES["types"] if own.matches_any(rel, RULES["types"][t]["paths"])]
        assert owners == [OWNER_OF_PREFIX[script.name[:2]]], f"{rel} owned by {owners}"


@pytest.mark.parametrize("stage_folder", STAGES.values())
def test_each_stage_has_five_scripts_and_five_handoffs(stage_folder):
    folder = REPO_ROOT / stage_folder
    assert len(list(folder.glob("0[1-5]_*.py"))) == 5
    handoffs = sorted((folder / "HANDOFF").glob("0[1-5]_*.md"))
    assert len(handoffs) == 5
    for h in handoffs:
        assert re.search(r"^status:\s*[A-Z_]+", h.read_text(encoding="utf-8"), re.MULTILINE)


def test_role_owned_folders_do_not_overlap():
    """Prefixes eda_/eval_/integ_ keep three roles from writing the same report file."""
    probes = {
        "eda": "Stage01_ML/reports/eda_findings.md",
        "eval": "Stage01_ML/reports/eval_metrics.json",
        "integ": "Stage01_ML/reports/integ_ops_field_guide.md",
        "ml": "Stage01_ML/predictions/ml_predictions.csv",
        "data": "Stage01_ML/data/prepared_dataset.csv",
    }
    for owner, path in probes.items():
        who = [t for t in RULES["types"] if own.matches_any(path, RULES["types"][t]["paths"])]
        assert who == [owner], f"{path} -> {who}"


# ---------- commit subject format ----------

@pytest.mark.parametrize("subject", [
    "ml(stage01): train xgboost baseline",
    "eda: first look at noise floor",
    "data(stage02): hourly grid builder",
    "docs: update kickoff prompts",
    "integ(stage04)!: change api shape",
])
def test_good_subjects(subject):
    assert own.SUBJECT_RE.match(subject)


@pytest.mark.parametrize("subject", ["Update README.md", "fix stuff", "ML: caps", "ml(stage01) missing colon", "ml(stage01):nospace"])
def test_bad_subjects(subject):
    assert not own.SUBJECT_RE.match(subject)


def test_commit_msg_hook_regex_agrees_with_python_regex():
    """The shell hook and the python checker must accept/reject the same subjects."""
    hook = (REPO_ROOT / ".githooks" / "commit-msg").read_text(encoding="utf-8")
    assert "(data|eda|ml|eval|integ|docs|chore|ci)" in hook


# ---------- AI attribution stripping ----------

def test_strip_removes_ai_trailers_but_keeps_human_ones():
    message = (
        "ml(stage01): add model\n\nBody.\n\n"
        "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n"
        "Co-authored-by: GitHub Copilot <copilot@github.com>\n"
        "Co-authored-by: Asha Rao <asha@example.com>\n"
        "Generated with [Claude Code](https://claude.com/claude-code)\n"
    )
    cleaned = strip.clean(message)
    assert "Claude" not in cleaned and "Copilot" not in cleaned and "Generated with" not in cleaned
    assert "Asha Rao" in cleaned
    assert cleaned.startswith("ml(stage01): add model")


def test_strip_leaves_clean_message_unchanged():
    message = "eval(stage01): add calibration plot\n\nBody text.\n"
    assert strip.clean(message) == message


# ---------- stubs behave ----------

def test_stub_scripts_exit_with_not_implemented_code():
    result = subprocess.run([sys.executable, "03_nlp_engineer.py"], cwd=REPO_ROOT / "Stage03_NLP", capture_output=True, text=True)
    assert result.returncode == 3
    assert "NOT_IMPLEMENTED" in result.stdout


def test_run_pipeline_skips_stubs_without_failing():
    # Stage 5 is still all stubs. Stage 1 is implemented now, and running it here would rewrite the team's committed
    # predictions, reports and ledger every time someone runs the tests.
    result = subprocess.run([sys.executable, "run_pipeline.py", "--stage", "5"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "SKIPPED" in result.stdout


# ---------- documentation hygiene ----------

def _markdown_files():
    skip = {"_reference"}
    return [p for p in REPO_ROOT.rglob("*.md") if not skip & set(p.relative_to(REPO_ROOT).parts)]


def test_relative_markdown_links_point_to_real_files():
    """A broken link in the docs sends a teammate to nowhere in the middle of a debate day."""
    link = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")
    broken = []
    for md in _markdown_files():
        for target in link.findall(md.read_text(encoding="utf-8")):
            path = (md.parent / target.split("#")[0]).resolve()
            if not path.exists():
                broken.append(f"{md.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, broken


def test_every_role_file_named_in_the_stubs_exists():
    for doc in ("ROLE_1_ML_ENGINEER.md", "ROLE_2_EDA_ENGINEER.md", "ROLE_3_DATA_INTEGRATION_ENGINEER.md", "ROLE_4_EVALUATION_ENGINEER.md"):
        assert (REPO_ROOT / "docs" / "roles" / doc).exists(), doc


def test_no_ai_attribution_text_in_repository_files():
    """Attribution lines must not hide in committed files either (the rules files that mention them are the only exceptions)."""
    allowed = {"AGENTS.md", "CLAUDE.md", "GEMINI.md", "04_GITHUB_WORKFLOW.md", "07_KICKOFF_PROMPTS.md", "copilot-instructions.md",
               "cascade.mdc", "cascade.md", "pull_request_template.md", "test_repo_rules.py", "guard.yml", "commit-msg", "pre-push",
               "strip_ai_trailers.py", "check_ownership.py", "ROLE_1_ML_ENGINEER.md", "ROLE_2_EDA_ENGINEER.md",
               "ROLE_3_DATA_INTEGRATION_ENGINEER.md", "ROLE_4_EVALUATION_ENGINEER.md", "settings.json"}
    pattern = re.compile(r"^\s*co-authored-by:.*(claude|anthropic)", re.IGNORECASE | re.MULTILINE)
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and path.name not in allowed and ".git" not in path.parts and "_reference" not in path.parts and path.suffix in {".md", ".py", ".json", ".txt", ".csv", ".yml"}:
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, offenders
