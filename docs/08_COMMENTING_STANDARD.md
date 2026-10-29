# 08 - Commenting standard (plain English, every step)

**Why:** in the debate any teammate can be asked about any file, and teachers read our code. A reader who has
never seen the project must understand *what each step does and why* without running anything. Comments are
part of the deliverable, not decoration.

**For the AI assistant reading this:** you must follow this standard in every file you create or edit. If a
teammate's file lacks comments, add them in a separate `docs`-neutral commit only if you own the file; otherwise
leave it alone.

## The five rules

1. **File header docstring** (top of every `.py` file): what the file does, what it reads, what it writes, who owns it.
2. **One-line docstring on every function and class** saying what it does in plain words.
3. **A comment above every logical step** inside a function (loading, cleaning, one transformation, one model fit, one plot, one save). Say *what* and, when it is not obvious, *why*.
4. **Explain numbers that are not obvious**: any threshold, hyperparameter or magic number gets a comment with its meaning and where it came from (e.g. `# 24 h = CRITICAL threshold from contracts/label_policy.json`).
5. **Write for a first-year student**: short sentences, no jargon without a hint. Prefer "keep only the training rows" over "apply the train mask".

## What good looks like

```python
"""ROLE: EDA ENGINEER | STAGE 01
Finds out how noisy each port's dwell time is and whether the reporting changed over time.

READS : Stage01_ML/data/prepared_dataset.csv
WRITES: Stage01_ML/reports/eda_noise_floor.csv, eda_drift_report.json
"""

from common.paths import stage_path
import pandas as pd


def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Return the typical random wobble (noise) of dwell time for every node."""
    # Sort by time so that neighbouring rows are neighbours in real life too.
    df = df.sort_values("timestamp")

    # For each node, smooth the dwell time with a 24-row rolling median.
    # The smooth line is the "signal"; what is left over is the "noise".
    smooth = df.groupby("current_node_id")["dwell_time_hours"].transform(
        lambda s: s.rolling(24, min_periods=6, center=True).median()
    )
    residual = df["dwell_time_hours"] - smooth

    # MAD (median absolute deviation) is a noise measure that ignores a few huge spikes,
    # so one strike day does not inflate the "normal" noise level.
    mad = residual.abs().groupby(df["current_node_id"]).median()

    # Multiply by 1.4826 so MAD is comparable to a standard deviation for bell-shaped noise.
    return (mad * 1.4826).rename("noise_floor_hours").reset_index()
```

## What bad looks like (do not do this)

```python
def nf(df):                                  # no docstring, cryptic name
    df = df.sort_values("timestamp")
    s = df.groupby("current_node_id")["dwell_time_hours"].transform(lambda s: s.rolling(24, min_periods=6, center=True).median())
    return ((df["dwell_time_hours"] - s).abs().groupby(df["current_node_id"]).median() * 1.4826).reset_index()   # no comments at all, magic numbers
```

## Rules for other file types

| File type | What to comment |
|---|---|
| `.py` scripts | Everything above. |
| SQL / config / YAML / JSON | JSON cannot hold comments: put a `"note"` or `"description"` key next to non-obvious values. YAML/SQL: use `#` / `--` comments. |
| Shell hooks / scripts | A header explaining purpose and a comment before every command block. |
| Streamlit pages | A comment above each widget group: "this block lets the operator pick a node". |
| Reports (`.md`) | Start with a 3-line "what this report shows and how to reproduce it" box including the exact command. |

## Anti-patterns the reviewer will flag

- Comments that only repeat the code (`x = x + 1  # add one to x`). Say *why* instead.
- Commented-out old code. Delete it; git remembers.
- Comments that lie after an edit. If you change the code, change the comment in the same commit.
- Hidden magic numbers with no explanation.
- Giant functions without step comments. Split them, or comment each block.
