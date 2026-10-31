"""Contract checker: proves a CSV matches the agreed schema in contracts/schemas/.

Why this file exists: four people on four devices only integrate cleanly if every file that is
handed over has exactly the agreed columns, types, ranges and allowed values. Run this before
you write "READY" in your HANDOFF file.

Command line:
    python scripts/validate_contract.py telemetry Stage01_ML/data/prepared_dataset.csv
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.paths import NODES_CSV, SCHEMAS_DIR

# Text values that count as boolean true / false when a CSV is read as plain text.
_TRUE = {"true", "1", "yes", "t"}
_FALSE = {"false", "0", "no", "f"}


@dataclass
class Violation:
    """One broken rule: which column, which rule, how many rows, and one example value."""

    column: str
    rule: str
    count: int
    example: str = ""

    def __str__(self) -> str:
        tail = f" e.g. {self.example!r}" if self.example != "" else ""
        return f"[{self.rule}] column '{self.column}': {self.count} row(s){tail}"


@dataclass
class Report:
    """Result of one validation run."""

    schema: str
    n_rows: int
    violations: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def load_schema(name: str) -> dict:
    """Load a schema by name. A schema may say "extends": "other" to reuse that schema's columns."""
    path = SCHEMAS_DIR / f"{name}.schema.json"
    if not path.exists():
        known = sorted(p.name.replace(".schema.json", "") for p in SCHEMAS_DIR.glob("*.schema.json"))
        raise FileNotFoundError(f"No schema named '{name}'. Known schemas: {known}")
    schema = json.loads(path.read_text(encoding="utf-8"))

    parent_name = schema.get("extends")
    if parent_name:
        parent = load_schema(parent_name)
        # Child columns replace parent columns that share a name; new ones are appended.
        merged = {c["name"]: c for c in parent["columns"]}
        for column in schema["columns"]:
            merged[column["name"]] = column
        schema["columns"] = list(merged.values())
    return schema


def load_node_ids() -> set:
    """Return the set of official node ids from contracts/network_nodes.csv."""
    return set(pd.read_csv(NODES_CSV)["node_id"])


def _sample(series: pd.Series) -> str:
    """First offending value, shortened, for readable error messages."""
    return str(series.iloc[0])[:60] if len(series) else ""


def _check_dtype(name: str, series: pd.Series, dtype: str) -> tuple:
    """Return (converted series, list of Violation) for the declared dtype."""
    present = series.notna()
    out: list = []

    if dtype == "string":
        return series, out

    if dtype in ("int", "float"):
        converted = pd.to_numeric(series, errors="coerce")
        bad = present & converted.isna()
        if bad.any():
            out.append(Violation(name, f"not_{dtype}", int(bad.sum()), _sample(series[bad])))
        if dtype == "int":
            fractional = converted.notna() & (converted != converted.round())
            if fractional.any():
                out.append(Violation(name, "not_int", int(fractional.sum()), _sample(series[fractional])))
        return converted, out

    if dtype == "bool":
        lowered = series.astype(str).str.strip().str.lower()
        bad = present & ~lowered.isin(_TRUE | _FALSE)
        if bad.any():
            out.append(Violation(name, "not_bool", int(bad.sum()), _sample(series[bad])))
        return lowered.isin(_TRUE).where(present), out

    if dtype == "datetime":
        converted = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
        bad = present & converted.isna()
        if bad.any():
            out.append(Violation(name, "not_iso_datetime", int(bad.sum()), _sample(series[bad])))
        return converted, out

    raise ValueError(f"Schema uses unknown dtype '{dtype}' for column '{name}'")


def validate_dataframe(df: pd.DataFrame, schema: dict, node_ids: set | None = None) -> Report:
    """Check a DataFrame against a loaded schema and return a Report."""
    report = Report(schema=schema["name"], n_rows=len(df))
    node_ids = node_ids if node_ids is not None else load_node_ids()
    declared = [c["name"] for c in schema["columns"]]

    # Columns we do not know about are allowed but listed, so nobody is surprised later.
    report.extra_columns = [c for c in df.columns if c not in declared]

    if len(df) < schema.get("min_rows", 0):
        report.violations.append(Violation("*", "too_few_rows", len(df), f"need >= {schema['min_rows']}"))

    converted: dict = {}
    for col in schema["columns"]:
        name = col["name"]
        required = col.get("required", True)

        # 1. The column must exist (unless it is marked optional).
        if name not in df.columns:
            if required:
                report.violations.append(Violation(name, "missing_column", len(df)))
            continue

        series = df[name]
        values, dtype_violations = _check_dtype(name, series, col["dtype"])
        report.violations.extend(dtype_violations)
        converted[name] = values
        present = series.notna() & (series.astype(str).str.strip() != "")

        # 2. Nulls are not allowed unless the column says so.
        if not col.get("nullable", False):
            nulls = ~present
            if nulls.any():
                report.violations.append(Violation(name, "null_not_allowed", int(nulls.sum())))

        # 3. Numeric range.
        if col["dtype"] in ("int", "float"):
            if "min" in col:
                low = values.notna() & (values < col["min"])
                if low.any():
                    report.violations.append(Violation(name, f"below_min_{col['min']}", int(low.sum()), _sample(series[low])))
            if "max" in col:
                high = values.notna() & (values > col["max"])
                if high.any():
                    report.violations.append(Violation(name, f"above_max_{col['max']}", int(high.sum()), _sample(series[high])))

        # 4. Allowed values list.
        if "allowed" in col:
            bad = present & ~series.astype(str).isin(col["allowed"])
            if bad.any():
                report.violations.append(Violation(name, "value_not_allowed", int(bad.sum()), _sample(series[bad])))

        # 5. Text pattern (the whole value must match).
        if "pattern" in col:
            regex = re.compile(col["pattern"])
            # Missing cells are not strings, so they simply count as "no match" (and are masked by `present`).
            matches = series.map(lambda v: isinstance(v, str) and bool(regex.fullmatch(v.strip())))
            bad = present & ~matches.astype(bool)
            if bad.any():
                report.violations.append(Violation(name, "pattern_mismatch", int(bad.sum()), _sample(series[bad])))

        # 6. Node ids must exist in the official registry.
        if col.get("fk") == "nodes":
            bad = present & ~series.astype(str).isin(node_ids)
            if bad.any():
                report.violations.append(Violation(name, "unknown_node_id", int(bad.sum()), _sample(series[bad])))

        # 7. Column-level uniqueness.
        if col.get("unique"):
            dup = series[present].duplicated(keep=False)
            if dup.any():
                report.violations.append(Violation(name, "duplicate_value", int(dup.sum()), _sample(series[present][dup])))

    # Table-level: the primary key combination must be unique.
    key = [k for k in schema.get("primary_key", []) if k in df.columns]
    if key and len(key) == len(schema.get("primary_key", [])):
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            report.violations.append(Violation("+".join(key), "duplicate_primary_key", int(dup.sum())))

    # Table-level: groups of probability columns must add up to 1.
    for group in schema.get("sum_to_one", []):
        if all(g in converted for g in group):
            total = sum(converted[g] for g in group)
            off = total.notna() & ((total - 1).abs() > 1e-3)
            if off.any():
                report.violations.append(Violation("+".join(group), "probabilities_do_not_sum_to_1", int(off.sum()), _sample(total[off].round(4))))

    return report


def write_validated_csv(df: pd.DataFrame, schema_name: str, path: str | Path) -> Report:
    """Validate a DataFrame against a schema, and only write the CSV when it passes.

    Use this in your role script for every hand-over file, so a broken file never reaches another device.
    Raises ValueError with the list of broken rules when validation fails.
    """
    report = validate_dataframe(df.astype(object).where(df.notna(), None), load_schema(schema_name))
    if not report.ok:
        details = "\n".join(f"  - {v}" for v in report.violations)
        raise ValueError(f"Refusing to write {path}: it breaks the '{schema_name}' contract:\n{details}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # A fixed gzip timestamp (mtime=0) makes a re-run with the same data give the same bytes, so git shows no change.
    compression = {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None
    df.to_csv(path, index=False, compression=compression)
    return report


def validate_csv(schema_name: str, path: str | Path) -> Report:
    """Read a CSV as plain text (so nothing is silently converted) and validate it."""
    schema = load_schema(schema_name)
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    return validate_dataframe(df, schema)
