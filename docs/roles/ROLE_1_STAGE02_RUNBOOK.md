# Role 1 (Santhosh) - Stage 2 DL runbook: full training run, studies, hand-off

> **What this is:** the exact, step-by-step instructions for finishing Role 1's Stage 2 work, written so that you AND
> your AI agent (Google Antigravity) can follow them without guessing. Follow the steps in order. Every command is
> copy-paste ready. Written by Role 4 (Bennish) on 31 Oct 2026, after Stage 2 was built end to end (PR #20,
> `docs/decisions/0011-stage02-forecasting-design.md`). Everything here stays inside Role 1's lane.

---

## 0. Read this first (the rules that matter most today)

1. **The code already works.** It is reviewed, tested (189 tests pass) and evaluated. Do NOT let the agent "rewrite",
   "clean up" or "modernise" `03_dl_engineer.py` or `common/modelkit/dl_*.py`. Only make the changes listed in this file.
2. **Your lane** (`contracts/ownership.json`, type `ml`). You may change:
   - `Stage02_DL/03_dl_engineer.py`, `Stage02_DL/artifacts/**`, `Stage02_DL/predictions/**`, `Stage02_DL/requirements.txt`
   - `Stage02_DL/HANDOFF/03_model_engineer.md`
   - `common/modelkit/**`, `tests/ml/**`, `notebooks/ml/**`

   **Never** touch `Stage02_DL/data/**`, the `01_*`, `02_*`, `04_*` or `05_*` scripts, `Stage02_DL/reports/**`, `claims/**`, `contracts/**`, `pages/**` or `app.py`.
3. **Never use the test split to decide anything.** All choices (hidden size, window, graph tweaks) are made on
   VALIDATION numbers only (`val_pinball_scaled` in `model_card.json`, or `hparam_log.csv`).
   **Do not run `04_evaluation_engineer.py` or open any `reports/eval_*` file while you are still choosing settings.**
   Role 4 runs the evaluation after your final run.
4. **Partial runs are safe, full runs are the hand-off.** Any run that uses `--models`, `--epochs` or `--track sim|real`
   writes ONLY to `Stage02_DL/artifacts/scratch/` and `Stage02_DL/predictions/scratch/`. Both are git-ignored.
   Only the plain full command in step 4 writes the files you commit.
5. **Seed 42 stays.** Never change `common/seed.py` or the `SEED` value.
6. **No AI attribution.** No `Co-authored-by`, no "Generated with", no robot lines in commits or PR text (AGENTS.md R1).
   The git hooks strip them, but check anyway.
7. **Long commands run in YOUR terminal, not inside the agent.** The full run takes hours. An agent's terminal tool
   can time out or be cancelled and kill it. Let the agent prepare the commands; you start them in a normal terminal window.

---

## 1. Recommendation: where and how to train

| Option | Verdict | Why |
|---|---|---|
| **Your own laptop CPU, overnight** | **Recommended** | The models are tiny (about 17k-40k weights in fast mode, a few times more in full mode). Batch preparation runs on the CPU anyway, and the code is CPU-only and tested on CPU |
| Google Colab / Kaggle GPU | Not recommended | There is no GPU code path (no `.to(device)`). The free Colab CPU is slower than a modern laptop, and syncing artefacts back adds risk |
| Running it inside the Antigravity agent | No | Its tool calls can time out (rule 7) |

- **Expected time:** see section 4. It is a few hours, so start it in the evening or at lunch.
- **Keep the laptop awake and plugged in:**
  - macOS: put `caffeinate -i` in front of the command (shown in step 4).
  - Windows: Settings -> System -> Power -> "When plugged in, put my device to sleep after" = Never.
- Close other heavy programs (browsers with many tabs, games). The script uses 4 CPU threads.

---

## 2. One-time setup (15 minutes)

All commands run from the repository root.

```bash
git checkout main
git pull --rebase origin main            # PR #20 (Stage 2 build) must be merged first; ask Bennish if not
git config core.hooksPath .githooks      # once per clone: safety hooks
git checkout -b ml/stage02-full-run      # your branch (AGENTS.md R8: <role>/<stage>-<topic>)
```

Python 3.12 is the tested version. 3.14 has no tested PyTorch build. Create a fresh environment:

macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r Stage02_DL/requirements.txt
```

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt -r Stage02_DL/requirements.txt
```

Check that everything imports, and that the prepared data is there. You do NOT need to run `01_data_engineer.py` or download PortWatch: Role 3's data files are already committed.
```bash
python -c "import torch, xgboost, pandas, sklearn; print(torch.__version__, xgboost.__version__, pandas.__version__)"
ls Stage02_DL/data
```
You must see `hourly_node_telemetry.csv.gz`, `scaler.json`, `splits.json`, `real_daily_activity.csv.gz`,
`real_scaler.json`, `real_splits.json` and the two adjacency CSVs.

macOS note: PyTorch and XGBoost each bring their own OpenMP library, and the two deadlock in one process.
`03_dl_engineer.py` and `tests/conftest.py` already set `OMP_NUM_THREADS=1` before importing them. **Keep that line**
in any new script or notebook that imports both.

---

## 3. Understand the code before you run it (1 hour; you will be asked to explain it)

Read in this order:
1. `docs/decisions/0011-stage02-forecasting-design.md`: why v1.0 was withdrawn and what replaced it.
2. `common/modelkit/dl_data.py`: `ForecastData.batch()` (x = last 72 h, ctx = 28 daily blocks, anchor = last value), `split_origins()` (the same split rule as `common/dataio/windows.py`), `transit_edges()` (lane delay -> which past block a message reads).
3. `common/modelkit/dl_models.py`:
   - quantile head ("last value + learnt change", q10 <= q50 <= q90);
   - `LSTMForecaster`, `TransformerForecaster`, `GraphForecaster` (LSTM + context + 2 GCN layers + transit-delayed messages; the head reads `[h, graph, messages]`).
4. `common/modelkit/dl_train.py`:
   - `train_neural` (AdamW, pinball loss, clip 1.0, early stopping on val);
   - the baselines;
   - `train_xgb_lag` (3 quantiles, one model per horizon x target);
   - `conformal_margins` (split-conformal on val);
   - `long_frame` (contract rows).
5. `Stage02_DL/03_dl_engineer.py`: `settings()`, `model_config()` (the three graph variants differ ONLY in lanes), `run_track()`.
6. `tests/ml/test_dl_data.py` and `tests/ml/test_dl_models.py`: what is proven (no future leakage, ordered quantiles, equal ablation size, save/load round trip).

You should be able to explain, in two sentences each:
- why changing every value after the origin must not change the inputs (the leakage test);
- why the ablations are trained separately (and why v1.0's "swap the matrix at prediction time" was wrong);
- what the transit-delayed message reads for a 14-day lane at horizon 24 h (answer: the source's state ~13 days ago);
- what the conformal step does to an interval that is too narrow.

Run the tests:
```bash
python -m pytest tests/ml -q
```
Expected: all pass. If a test fails before you changed anything, stop and tell Bennish.

---

## 4. [MUST] The full training run

### 4.1 Smoke test (2-3 minutes, writes to scratch only)
```bash
cd Stage02_DL
python 03_dl_engineer.py --track sim --models persistence lstm --epochs 1
```
Expected: it prints `PARTIAL RUN: outputs go to ... scratch/` and ends with `Stage 02 models done`.
`git status` must show NO changes under `Stage02_DL/artifacts` or `Stage02_DL/predictions`.

### 4.2 The real thing (one command, both tracks, all 9 models)
macOS / Linux:
```bash
cd Stage02_DL
caffeinate -i python -u 03_dl_engineer.py 2>&1 | tee ~/stage02_full_run.log
```
Windows (PowerShell):
```powershell
cd Stage02_DL
python -u 03_dl_engineer.py 2>&1 | Tee-Object $HOME\stage02_full_run.log
```
- Settings (automatic, no flags): hidden 64, up to 30 epochs, early-stopping patience 5, training origins every 2nd hour (sim) / every day (real), XGBoost up to 600 trees. `model_version` will be `v2.0` (no `-fast`).
- **Expected duration:** up to about 2 h 15 min on an Apple M-series laptop, 3-4 h on a slower one (measured epoch times are below). Early stopping usually ends models well before 30 epochs.
- Do NOT add `--fast`, `--epochs` or `--models`: with any of those flags the outputs go to scratch, by design.
- If it crashes or the laptop sleeps, run the same command again. It starts from scratch and overwrites cleanly.

**Measured on Bennish's laptop (Apple M-series, 4 threads), one epoch at full settings:**

| Track | Model | Weights (hidden 64) | One epoch |
|---|---|---|---|
| sim (2,953 training origins) | lstm / transformer / graph_* | 60,870 / 78,406 / 90,406 | about 45 s each (3 models in 2.2 min, prediction included) |
| real (1,792 training origins) | lstm / transformer / graph_* | 58,835 / 74,899 / 81,235 | 8 s / 11 s / 6 s |

Worst case (all 5 neural models x 30 epochs): sim ~1 h 50 min + real ~20 min + XGBoost ~5 min, **about 2 h 15 min on
that laptop**. Early stopping usually ends sooner. On a slower laptop, plan for 3-4 h.

### 4.3 What a healthy run prints
- `=== sim track: 27 nodes, 8760 steps of 1 h ...`, then `origins: train ~2,960 (stride 2), val 149, test 151`
- `training xgb_lag ...` with 10 lines `xgb_lag dwell h=... best iteration ...`
- For each of lstm, transformer, graph_full, graph_shuffled, graph_none: `training <name> (<N> parameters) ...`, epoch lines, usually `early stop: no val improvement for 5 epochs`.
- The three graph models must show the **same** parameter count.
- `=== real track: 23 nodes, 2762 steps of 24 h ...`, then the same models.
- `wrote 729,000 rows -> .../dl_forecasts.csv.gz` and `wrote 530,334 rows -> .../dl_real_forecasts.csv.gz`
- `Stage 02 models done in N min. Champion proposal (val): <model>`

### 4.4 Check the outputs (from the repository root)
```bash
python scripts/validate_contract.py dl_forecasts Stage02_DL/predictions/dl_forecasts.csv.gz
python -c "import json; c=json.load(open('Stage02_DL/artifacts/model_card.json')); print(c['model_version'], c['run_seconds'], c['champion_proposal']); [print(t, m, v.get('parameters'), v.get('epochs_run'), v.get('best_epoch'), v.get('val_pinball_scaled')) for t, d in c['tracks'].items() for m, v in d['models'].items()]"
ls -la Stage02_DL/artifacts Stage02_DL/predictions
python -m pytest tests -q
```
Must hold:
- `PASS ... (729000 rows ...)`;
- `model_version` = `v2.0`;
- every file < 50 MB (the XGBoost `.joblib` files are the biggest, ~7-20 MB);
- all tests pass.

Then look at `Stage02_DL/artifacts/training_curve.csv`:
- `val_pinball` should fall and then flatten;
- if a model's `best_epoch` equals the last epoch run (30), it was still improving. Write that in your HANDOFF note.

---

## 5. [SHOULD] Hyper-parameter study (validation only)

Goal (role file): hidden 32 / 64 / 128 x window 48 / 72 / 168, logged to `Stage02_DL/artifacts/hparam_log.csv`.

### 5.1 Code change (ask the agent to implement exactly this, in `03_dl_engineer.py` + `common/modelkit/dl_data.py`)
1. New flags:
   - `--hidden N` overrides `settings()["hidden"]`;
   - `--input-len N` (sim track only) overrides the input window;
   - `--tag TEXT` labels the run.

   Any of them makes the run a partial run (scratch outputs), like `--models`.
2. `--input-len`: give `sim_data_from_grid(...)` an optional `spec` argument and pass `dataclasses.replace(SIM_TRACK, input_len=N)`. Pass `input_len` into the model config, so the Transformer's position table has the right size.
3. **Fair comparison:** after training each neural model, compute its val pinball on ONE common set of origins: the val origins that are valid for the LONGEST window, from `split_origins(times, dataclasses.replace(SIM_TRACK, input_len=168), bounds, origin_hours=(0, 6, 12, 18))["val"]`. Origins valid for 168 h are also valid for 48 and 72 h. Without this, longer windows would be scored on fewer, different days.
4. Append one row per model to `Stage02_DL/artifacts/hparam_log.csv` (create it with a header if missing; this file is NOT in scratch, because it is a hand-off file). Columns: `utc_time, tag, track, model, hidden, input_len, epochs_run, best_epoch, val_pinball_common, parameters, train_seconds, train_stride, git_commit`.
5. Add a unit test in `tests/ml/` that checks the common origin set is the same for input lengths 48, 72 and 168.
6. Plain-English comments on every step (docs/08_COMMENTING_STANDARD.md).

### 5.2 Runs (fast stride for speed; the same settings for every row)
```bash
cd Stage02_DL
for H in 32 64 128; do for L in 48 72 168; do
  python 03_dl_engineer.py --fast --track sim --models lstm graph_full --hidden $H --input-len $L --tag hp_${H}_${L}
done; done
```
(Windows: run the 9 combinations one by one, or ask the agent for a PowerShell loop.) About 9 x 5-8 minutes.

### 5.3 Decide (on validation only) and report
- Pick the lowest `val_pinball_common` per model. Write the table and the decision in your HANDOFF note.
- **Hidden size** may change in the committed models: it is saved inside each `.pt`, and the API reads it from there. If you change the default, edit `settings()`, then redo step 4.
- **The window stays 72 h in the committed models**, even if 168 h wins. The API (`POST /forecast`) and `dl_infer.py` take exactly 72 hours (role file and Role 3's contract with users). If a longer window wins clearly, write it as a proposal for Role 3 and Role 4 in your HANDOFF "Open questions". Do not change it alone.

---

## 6. [SHOULD] Try to make the graph model help (validation only, optional)

Fact from Role 4's evaluation of the fast run: graph_full is not better than graph_none or graph_shuffled (both tracks), and the plain LSTM / Transformer beat all graph variants. Role 2's lead-lag analysis will show whether the transit lags are even visible in the data. You may try, **judging only by `val_pinball_common` from step 5**:
- `gcn_layers=1`, or 0 (messages only);
- dropout on the messages;
- `msg_dim` 16;
- removing the context path (`ctx_enc`) from graph_none / graph_full.

Rules:
- Every variant gets a `--tag` row in `hparam_log.csv`.
- Whatever you change must apply **identically to all three graph variants**, so the ablation stays fair.
- Keep the change only if graph_full improves on val. Then redo step 4.
- A negative graph result is acceptable and will be reported honestly. Do not keep trying until a test number looks good (you are not allowed to look at test numbers anyway).

---

## 7. [SHOULD] Rolling-origin backtest (`--fold k`) for Role 4

Goal: skill spread across 3 earlier periods, all inside the TRAIN period, so test and val stay untouched.

Implement in `03_dl_engineer.py` (sim track only):
1. `--fold k` with k in {1, 2, 3}. Cutoffs (UTC): `C1 = 2025-06-01`, `C2 = 2025-07-01`, `C3 = 2025-08-01`.
2. Fold boundaries:
   - `train_end = C_k`
   - `val_start = C_k + 168 h` (purge)
   - `val_end = val_start + 21 days`
   - `test_start` = a date after the data (no test origins)

   Pass this dict to `split_origins()` instead of `splits.json`. All fold-val windows end before 2025-09-13, the real train_end.
3. **Refit the scaler for the fold** with `common.dataio.hourly_grid.fit_robust_scaler(grid, pd.Timestamp(C_k))`, so no scaling statistic comes from after the cutoff.
4. Models: persistence, seasonal_naive, xgb_lag, lstm, transformer, graph_full, graph_none. Use fast settings (`--fast`). Early stopping on the fold-val origins. No conformal step (Role 4 uses the point forecast for skill spread).
5. Predictions on fold-val origins every 6 h (clock hours 0/6/12/18), in the `dl_forecasts` columns with `split = "val"`, `model_version = "v2.0-fold-fast"` and an extra column `fold`. Append all three folds into `Stage02_DL/predictions/dl_backtest_folds.csv.gz` (a hand-off file, not scratch). Do not save model weights for folds.
6. Run:
   ```bash
   for K in 1 2 3; do python 03_dl_engineer.py --fast --fold $K; done
   ```
7. Tell Role 4 the file is ready. Role 4 adds the reading code to `04_evaluation_engineer.py` (your file must not compute skill itself).

---

## 8. [STRETCH] only if 4-7 are done
- 1D-CNN baseline: `CNNForecaster` in `dl_models.py` (dilated `Conv1d` stack, kernel 3, dilations 1/2/4/8, node embedding, the same `QuantileHead`), registered in `MODEL_CLASSES` and in `NEURAL` / `ARTIFACT_NAMES` in `03_dl_engineer.py`, plus shape/order/save-load tests like the existing ones. The evaluation picks up new model names automatically. Tell Role 3 if the API should serve it (`MODELS` dict in `05_integration_engineer.py`).
- Per-node-type heads: one `QuantileHead` per node type (PORT / CHOKEPOINT / RAIL_HUB / DC).

---

## 9. [MUST] Update your hand-off note (`Stage02_DL/HANDOFF/03_model_engineer.md`)

Change these fields:
- `status: READY`, `updated_utc: <today>`, `git_commit: <your commit hash after step 10>`.
- "Files produced": the row counts from the log; the artefact sizes; `model_version v2.0`; `hparam_log.csv` and `dl_backtest_folds.csv.gz` if done.
- "Contract check": paste the exact output of `python scripts/validate_contract.py dl_forecasts Stage02_DL/predictions/dl_forecasts.csv.gz`.
- "Decisions": full-run settings, training time (`run_seconds`), `best_epoch` per model, the hparam decision (validation numbers only), any graph change and why.
- "Things the next role must know": remove "Current files are from --fast mode". Keep the macOS OpenMP note.
- Delete the "Your tasks (Santhosh)" list, or tick off what is done and say what is not.

---

## 10. [MUST] Commit, push, PR

```bash
git status
```
Add ONLY these (your lane):
```bash
git add Stage02_DL/artifacts/*.pt Stage02_DL/artifacts/*.joblib Stage02_DL/artifacts/model_card.json Stage02_DL/artifacts/training_curve.csv
git add Stage02_DL/predictions/dl_forecasts.csv.gz Stage02_DL/predictions/dl_real_forecasts.csv.gz
git add Stage02_DL/HANDOFF/03_model_engineer.md
# only if you did steps 5-8:
git add Stage02_DL/artifacts/hparam_log.csv Stage02_DL/predictions/dl_backtest_folds.csv.gz Stage02_DL/03_dl_engineer.py common/modelkit tests/ml
```
**Never add:**
- `Stage02_DL/data/**`, `Stage02_DL/reports/**`, `claims/**`, `data/calibration/**`;
- the `01_*`, `04_*` or `05_*` scripts;
- anything in `scratch/`, any `.log` file, `.venv/`.

If `git status` shows other files changed that you did not mean to change, ask before committing.

Commit (one idea per commit; examples):
```bash
git commit -m "ml(stage02): full training run (hidden 64, up to 30 epochs) for both tracks"
git commit -m "ml(stage02): hyper-parameter study on validation (hparam_log.csv)"
git commit -m "ml(stage02): rolling-origin backtest folds for Role 4"
```
Then:
```bash
python scripts/check_ownership.py --range origin/main..HEAD     # must print OK
python -m pytest tests -q                                       # must pass
git pull --rebase origin main
git push -u origin ml/stage02-full-run
```
Open a PR to `main` using the template (title like `ml(stage02): full training run + hparam study`), with no AI attribution in the title or body. Then message Bennish ("full run is on the branch") and Pravin (the API serves the new `.pt` files).

---

## 11. Driving Antigravity (copy this as the FIRST message of a new agent conversation)

```
I am Role 1, the ML Engineer (scripts 03_* in every stage). Today: Stage 2 DL, finishing work.
1. Read AGENTS.md and obey it. Then read docs/roles/ROLE_1_ML_ENGINEER.md (Day 2 section) and
   docs/roles/ROLE_1_STAGE02_RUNBOOK.md completely. The runbook is my task list: follow it step by step, in order.
2. Stage 2 code already works and is tested. Do NOT rewrite, refactor or "improve" existing code. Only make the code
   changes the runbook describes (sections 5.1, 6, 7, 8), exactly as specified, with plain-English comments and tests.
3. Edit only Role 1 files: Stage02_DL/03_dl_engineer.py, Stage02_DL/artifacts/**, Stage02_DL/predictions/**,
   Stage02_DL/HANDOFF/03_model_engineer.md, common/modelkit/**, tests/ml/**. Never touch contracts/, Stage02_DL/data/,
   reports/, claims/, or the 01/02/04/05 scripts.
4. Never use test-split numbers for any decision. Do not run 04_evaluation_engineer.py.
5. Do NOT start long-running commands yourself (the full training run takes hours). Give me the exact command and
   I will run it in my own terminal. Short commands (pytest, validate_contract, git status) you may run after asking.
6. Keep os.environ.setdefault("OMP_NUM_THREADS", "1") before importing torch/xgboost in any script.
7. Never add Co-authored-by, "Generated with" or any AI attribution to commits or PRs. Commit subject: ml(stage02): message.
Start with runbook section 2 and tell me what you find.
```

Working tips:
- Use the agent's **planning mode** for code changes (sections 5-8). Read its plan before it edits, and review every diff. Reject edits outside your lane or rewrites of working code.
- Set terminal commands to **require your approval**. Never approve `git push --force`, `git reset --hard`, deleting `data/` or `artifacts/`, or changes to `contracts/`.
- One task per conversation: (a) setup + tests, (b) full run, (c) hparam flags, (d) fold backtest, (e) hand-off + PR. A short, focused context gives better results.
- After each code change: `python -m pytest tests/ml -q` before anything else.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Process hangs at 0% CPU, or a segfault right after XGBoost | OpenMP clash (torch + xgboost) | Make sure `OMP_NUM_THREADS=1` is set before the imports (it is in `03_dl_engineer.py`); never import torch after xgboost without it |
| `ModuleNotFoundError: torch` | Wrong Python, or the venv is not active | `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`); then `pip install -r Stage02_DL/requirements.txt` |
| `The hourly grid has gaps; run 01_data_engineer.py first` | Data files missing or old | `git pull` (the data is committed). Do not regenerate it yourself; ask Pravin |
| `[real] skipped: no prepared data` | `real_daily_activity.csv.gz` missing | `git pull`; ask Pravin. Do not download PortWatch yourself (that script also rewrites `data/calibration`) |
| `PARTIAL RUN ... scratch` when you wanted the real run | You added `--fast`/`--epochs`/`--models`/`--track` | Run the plain command from 4.2 (`--fast` alone is not partial, but it is not the full run either) |
| `Refusing to write ... breaks the 'dl_forecasts' contract` | A change produced bad rows (NaN, a wrong column) | Read the listed violations; fix the code; never edit the contract |
| The pre-commit hook blocks a file > 50 MB | Usually the XGBoost joblib with many trees | Tell Bennish; options: fewer trees on val, or `compress=9` in `joblib.dump` |
| `check_ownership` fails | A file outside your lane is staged | `git restore --staged <file>`; if you really need that change, ask its owner |
| Laptop slept; the run stopped | Power settings | Rerun 4.2 with `caffeinate -i` (macOS) or "never sleep" (Windows) |

---

## 13. Done checklist
- [ ] Section 3 understood; `pytest tests/ml` green
- [ ] Full run finished; `model_version v2.0`; contract PASS (729,000 rows); `dl_real_forecasts.csv.gz` 530,334 rows
- [ ] All artefacts < 50 MB; `training_curve.csv` checked
- [ ] (SHOULD) `hparam_log.csv` with 9+ rows and a validation-only decision
- [ ] (SHOULD) `dl_backtest_folds.csv.gz` with folds 1-3
- [ ] HANDOFF `03_model_engineer.md` = READY with the contract output pasted
- [ ] `check_ownership` OK, `pytest tests -q` green, PR opened, Bennish and Pravin told
