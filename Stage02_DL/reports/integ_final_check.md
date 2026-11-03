# Stage 2 - final integration check (Role 3, Pravin)

Date: 2026-11-02. Every result below comes from running the command shown; nothing is typed from memory.
- **Local repo** = this checkout on `main`.
- **Fresh clone** = a new `git clone` of `main` at `b4d26df` (after PR #28) in a temporary folder, with a new `.venv` built
  from `requirements-lock.txt` (Python 3.12.13).
- No model was retrained; `03_dl_engineer.py` was not run.

**Verdict: Stage 2 Integration is READY.** Every Stage 2 script reproduces its committed outputs from a fresh clone, the
close-out fixes are merged, all tests pass, and the live app shows the ensemble and the validated early-warning rule.
Stage 1 runs end to end but its numbers move slightly with library versions (row 10, below); that is documented, not hidden.

## Checklist

| # | Step | Command | Result | Verdict |
|---|---|---|---|---|
| 0 | Setup and board | `git pull --rebase origin main`; `python scripts/board.py` | `main` up to date; Stage02_DL: Data READY, EDA WIP, Model READY, Eval READY, Integ WIP | PASS |
| 1 | PR #27 on main + tests | `python -m pytest tests -q` on `7cb7c8d` | PR #27 already merged by Role 4; `230 passed, 23 skipped` (skips = in-process duplicates of checks run in a fresh process) | PASS |
| 2 | Review of Role 4's work in my lane | read `05_integration_engineer.py`, `common/modelkit/dl_infer.py` (EnsembleForecaster), `app.py`, `pages/0..2`, `tests/integ/*` | Ensemble served as default, pages read files only (no PyTorch), ensemble test checks the API against the file. One change needed: the Forecasts list used an untested 50% rule (fixed in step 4) | PASS |
| 3 | Fix 1: API fallback | `pytest tests/integ/test_api_stage02.py` | Default model outside the stored grid -> transformer, `evidence.model_fallback` set; explicit `?model=ensemble` -> 422 with hint. API tests `14 passed` | PASS |
| 4 | Fix 2: validated alert rule on the Forecasts page | `pytest tests/integ/test_control_tower_pages.py` | "Early warnings (next 7 days)": max P(dwell >= alert line) over 1 h-7 d >= `alert_rule.p_star` read from `eval_champion.json` (0.65); "Most likely when" column; new test recomputes the rule from the files and matches the page. `4 passed` | PASS |
| 5 | Fix 3: lock file | `pip freeze > requirements-lock.txt`; `pip install --dry-run -r requirements-lock.txt` | lock = tested environment exactly; the fresh clone's `.venv` was built from it and every test passed | PASS |
| 6 | Dashboard (local) | `streamlit run app.py` | Home: ensemble named, status counts. Shipments (ML): form scored a shipment (At risk 75%), status filter (1,785 At risk), row details. Forecasts (DL): ensemble named, p* sentence (65%), chart with history, forecast, 80% range, alert line. No error on screen | PASS |
| 7a | Ship | PR #28 | merged into `main` as `b4d26df`; `guard` checks passed; `main`: `233 passed, 23 skipped` | PASS |
| 7b | Live app | https://aipoweredpredictionofglobalsupplychainlogisticsbottlenecks-fut.streamlit.app/ | Home and Shipments (ML) load with no error; Forecasts (DL) shows the ensemble, "Early warnings (next 7 days)" with the 65% p* sentence and "Most likely when" | PASS |
| 7c | Live app after PRs #30 and #31 (3 Nov) | same URL; `pytest tests -q` on `6177a75` | Two pages, no home page. Shipments: "Check a shipment" scores the default shipment (At risk 75%). Port forecasts: ensemble and 65% p* named; opens on 19 Nov 2025 18:00 with 1 congestion warning (Busan 78%), 0 congested now, 26 normal; "All ports" table. No error; no reboot needed. Opening date and count re-computed independently from the files (below) | PASS |
| 8a | Real data (fresh clone) | `python scripts/fetch_real_data.py` | exit 0, 27 min 38 s; `data/calibration/*` IDENTICAL to committed | PASS |
| 8b | Data (fresh clone) | `cd Stage02_DL && python 01_data_engineer.py` | 236,520 rows (2,379 imputed); real track 23 nodes x 2,762 days; all 15 `Stage02_DL/data/*` + `data/calibration/*` files IDENTICAL. Contracts: `PASS hourly_node_telemetry (236520 rows)`, `PASS dl_forecasts (810000 rows)` | PASS |
| 8c | EDA (fresh clone) | `python 02_eda_engineer.py` | exit 0 (14 min 46 s under load); all 16 `eda_*` csv/json/md IDENTICAL; noise-channel test **PASS** | PASS |
| 8d | Evaluation (fresh clone, dry run) | `python 04_evaluation_engineer.py --out-dir <tmp>` | exit 0; trust checks PASS (10 models); all 20 `eval_*` csv/json/md IDENTICAL; `claims/CLAIMS_LEDGER.md` untouched | PASS |
| 8e | API (fresh clone) | `python 05_integration_engineer.py`; `pytest tests -q` | exit 0, 200 calls x 6 models, 0 failures; tests `233 passed, 23 skipped`, API tests `14 passed`. Timings under load not comparable (see latency) | PASS |
| 8f | Stage 1 (fresh clone) | `python run_pipeline.py --stage 1` | finishes (01-04 DONE, 8 min 43 s). 23 files IDENTICAL, 24 DIFFERENT: data equal in value, model numbers move slightly (below) | DIFFERENT - library versions |
| 9 | Lanes | `python scripts/check_ownership.py --range origin/main..HEAD` | OK | PASS |

PNG files are not compared byte for byte (image bytes may differ for identical charts).

## Latency over HTTP

`05_integration_engineer.py`: 200 `POST /forecast` calls per model through FastAPI's in-process TestClient, real 72-hour
test-period windows, 1 warm-up call not counted. In-process = no network hop; a deployed API adds network time.

| model | p95 ms, quiet laptop (committed, PR #28) | p95 ms, fresh clone (load average 36) |
|---|---|---|
| ensemble (default) | 31.52 | 160.45 |
| transformer | 7.98 | 61.36 |
| lstm | 11.06 | 43.76 |
| graph_full | 27.09 | 124.77 |
| graph_shuffled | 32.17 | 118.66 |
| graph_none | 27.10 | 120.44 |

The fresh-clone run shared the CPU with an unrelated training job (load average 36-72), so only its "0 failures" is
evidence; the committed numbers were measured when the laptop was quiet (load about 3).

## Early warnings vs ports already congested (step 7c)

Since PR #31 the Port forecasts page splits Role 4's rule B (highest P(dwell >= alert line) over 1 h-7 d >= p* = 0.65)
into **Congested now** (the measured waiting time is already above the line) and **Congestion warning** (not above it
yet). Counted independently from `dl_forecasts.csv.gz`, `hourly_node_telemetry.csv.gz` and `eval_alert_thresholds.csv`
(ensemble, test period, 151 forecast times, 27 nodes):
- rule B fires for at most 4 nodes at one time (22 Nov 2025 00:00), which is the "4 early warnings" shown before PR #31;
- **94% of rule-B firings are at a port that is already above its line**, so they are not early warnings;
- true early warnings (rule fires, not yet congested): at most 1 at a time (19 Nov 2025 18:00, the page's opening date),
  at 14 of the 151 forecast times.

The page is correct; this is a property of the rule, not a page bug. It is input for open item 3 (alert-rule choice):
rule B mostly confirms congestion that is already measured. Nothing was tuned on test numbers.

## Stage 1 details (step 8f)
- `Stage01_ML/data/prepared_dataset.csv.gz` (Role 3): same shape (57,718 x 35), no text differences, largest numeric
  difference 7.1e-15 (last-digit rounding). Same data in value.
- Champion stays logistic regression; its threshold moves 0.39 -> 0.40. Macro-F1: logreg 0.6658 -> 0.6660, xgboost
  0.6798 -> 0.6836, random_forest 0.6980 -> 0.6956, lightgbm/majority/rule_dwell unchanged, xgboost_monotone 0.6822 -> 0.6820.
- Cause: the committed Stage 1 model was built with Python 3.13.14, numpy 2.3.4, pandas 2.3.3, scikit-learn 1.8.0, xgboost
  3.1.3 (`Stage01_ML/artifacts/model_card.json`); the lock file pins the tested environment (scikit-learn 1.9.1, xgboost
  3.4.1). Known limitations row 10. Stage 1 EDA also reads the prepared file when it exists, so `eda_findings.md` says
  "prepared" in a pipeline run and "raw" in the committed one.

## Open items (owner)
1. Review the EDA re-run by Role 4 and set `Stage02_DL/HANDOFF/02_eda_engineer.md` READY - **Isha (Role 2)**.
2. Review the ensemble and the `--seeds` code (`common/modelkit/dl_infer.py`, `dl_train.py`, `03_dl_engineer.py`) - **Santhosh (Role 1)**.
3. Alert-rule choice, p95 rule vs the structural rule (`eda_episode_rule_check.csv`) - **Isha (Role 2) + Bennish (Role 4)**.
   New input (3 Nov): 94% of rule-B firings in the test period are at ports already above their line (section above).
4. A second simulated year for more bottleneck episodes is **not planned** (team decision, known limitations row 13).
5. Add `requirements-lock.txt` to the `integ` paths in `contracts/ownership.json` (its commit carries a Cross-role line) - **contracts owner, agreed by the team lead**.
6. Exact Stage 1 reproduction on the current libraries needs the Stage 1 model re-saved with the locked versions and the
   Stage 1 claims re-verified - **Role 1 + Role 4, only if the team wants it** (no retraining in this close-out).
7. After any change to `common/` is merged, reboot the Streamlit Cloud app: on 2 Nov the live app kept the old
   `common/integration/ui.py` in memory until a reboot - **app owner (Bennish), reminder for everyone**.
