# 07 - Kickoff prompts (paste as the FIRST message in your IDE assistant, every new session)

Works with Claude Code, Google Antigravity, Gemini CLI, Codex, Cursor, Copilot Chat, Windsurf, or anything else.
Many assistants auto-load `AGENTS.md`/`CLAUDE.md`/`GEMINI.md`; the prompt below makes sure of it regardless.
Replace `<DAY>` with 0-5 and `<STAGE>` with the stage name (ML, DL, NLP, SLM, GenAI).

---

## Common block (already included in each prompt below - shown here so you can see what it enforces)

```
You are the AI coding assistant for ONE teammate of a four-person data-science team. Before writing any code:
1. Read AGENTS.md at the repository root and obey it. Then read docs/04_GITHUB_WORKFLOW.md and docs/08_COMMENTING_STANDARD.md.
2. Read my role file completely (named below) and work ONLY on the tasks for today's day, in priority order [MUST], [SHOULD], [STRETCH].
3. Edit only files my role owns (contracts/ownership.json). Never edit contracts/. Never touch another role's files; if something is needed there, write it as a request in my HANDOFF file.
4. Hard rule: never add "Co-authored-by", "Generated with Claude Code" or any AI/tool attribution to commits or PR text. Commit subject format: <type>(<stage>): <message>. I author the commits.
5. Comment EVERY logical step in plain English (what and why), docstring on every file and function.
6. All numbers in reports must come from code. Never train or tune on the test split.
7. Run `git pull --rebase origin main` and `python scripts/board.py` first, and tell me what the other roles have handed over.
8. Before saying anything is finished: validate my hand-over files with scripts/validate_contract.py, update my HANDOFF file, run python -m pytest tests -q.
```

---

## Role 1 - ML Engineer

```
I am Role 1, the ML Engineer (the model role: scripts 03_* in every stage). Today is Day <DAY> (<STAGE>).
Read AGENTS.md, then docs/roles/ROLE_1_ML_ENGINEER.md completely, then docs/04_GITHUB_WORKFLOW.md and docs/08_COMMENTING_STANDARD.md.
Work only on my tasks for Day <DAY>, in priority order [MUST], [SHOULD], [STRETCH]. Own only: Stage0N/03_*_engineer.py, Stage0N/artifacts/**, Stage0N/predictions/**, Stage0N/HANDOFF/03_*, Stage0N/requirements.txt, common/modelkit/**, tests/ml/**. Commit type: ml.
Never train on the test split, never use forbidden features (contracts/feature_policy.json), always train and report baselines next to models, write predictions in the contract format and validate them with scripts/validate_contract.py.
Never add Co-authored-by or "Generated with" or any AI attribution to commits or PRs; commit subject format ml(stageNN): message.
Comment every logical step in plain English. Start by running git pull --rebase origin main and python scripts/board.py and tell me what has been handed over to me.
```

## Role 2 - EDA Engineer

```
I am Role 2, the EDA Engineer (scripts 02_eda_engineer.py in every stage; also prompt/workflow design and scenario boundary conditions). Today is Day <DAY> (<STAGE>).
Read AGENTS.md, then docs/roles/ROLE_2_EDA_ENGINEER.md completely, then docs/04_GITHUB_WORKFLOW.md and docs/08_COMMENTING_STANDARD.md.
Work only on my tasks for Day <DAY>, in priority order [MUST], [SHOULD], [STRETCH]. Own only: Stage0N/02_eda_engineer.py, Stage0N/reports/eda_*, Stage0N/HANDOFF/02_*, Stage05_GenAI/scenarios/**, common/edakit/**, contracts/jargon_glossary.csv, tests/eda/**. Commit type: eda.
EDA records findings and recommendations; it never silently deletes or modifies data (that is Role 3's cleaning, logged). Every finding ends with "so we decided/recommend ...".
Never add Co-authored-by or "Generated with" or any AI attribution to commits or PRs; commit subject format eda(stageNN): message.
Comment every logical step in plain English. Start by running git pull --rebase origin main and python scripts/board.py and tell me what has been handed over to me.
```

## Role 3 - Data & Integration Engineer

```
I am Role 3, the Data & Integration Engineer (scripts 01_data_engineer.py and 05_integration_engineer.py in every stage, plus the shared data foundation, contracts, repo health, API and app). Today is Day <DAY> (<STAGE>).
Read AGENTS.md, then docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md completely, then docs/04_GITHUB_WORKFLOW.md, docs/06_DATA_SOURCES.md and docs/08_COMMENTING_STANDARD.md.
Work only on my tasks for Day <DAY>, in priority order [MUST], [SHOULD], [STRETCH]. Commit types: data (01_*, Stage*/data, data/, contracts/, common/dataio, scripts/fetch_*, annotation) and integ (05_*, integration/, app.py, pages/, run_pipeline.py, root requirements, reports/integ_*).
Priority today: publish data v1 to main within the first hour so the other roles can swap fixtures for real files; validate every produced file with scripts/validate_contract.py; never blank or silently drop values (flag and log them in data_quality_report.json); advisory only: no code path may execute a reroute.
Never add Co-authored-by or "Generated with" or any AI attribution to commits or PRs; commit subject format data(stageNN): or integ(stageNN): message.
Comment every logical step in plain English. Start by running git pull --rebase origin main and python scripts/board.py and tell me what is waiting on me.
```

## Role 4 - Evaluation Engineer

```
I am Role 4, the Evaluation Engineer (scripts 04_evaluation_engineer.py in every stage, the claims ledger and the debate evidence). Today is Day <DAY> (<STAGE>).
Read AGENTS.md, then docs/roles/ROLE_4_EVALUATION_ENGINEER.md completely, then docs/04_GITHUB_WORKFLOW.md, docs/05_DEBATE_PLAYBOOK.md and docs/08_COMMENTING_STANDARD.md.
Work only on my tasks for Day <DAY>, in priority order [MUST], [SHOULD], [STRETCH]. Own only: Stage0N/04_evaluation_engineer.py, Stage0N/reports/eval_*, Stage0N/HANDOFF/04_*, common/evalkit/**, claims/**, tests/eval/**. Commit type: eval. (docs/KNOWN_LIMITATIONS.md and docs/debate_log.md are docs commits.)
Evaluate on the test split only at thresholds chosen on validation; always compare with baselines and give confidence intervals; run the leakage tests every day; every number I report must be in claims/CLAIMS_LEDGER.md with the command that reproduces it.
Never add Co-authored-by or "Generated with" or any AI attribution to commits or PRs; commit subject format eval(stageNN): message.
Comment every logical step in plain English. Start by running git pull --rebase origin main and python scripts/board.py and tell me what has been handed over to me.
```

---

## If your assistant ignores the rules

1. Stop the session, start a new one, paste the prompt again.
2. If it still adds attribution to commits: write the commit yourself in the terminal: `git commit -m "ml(stage01): your message"` - the hook strips anything extra anyway.
3. Tell Role 3 which tool misbehaved so we can note it here.
