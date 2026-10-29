"""Tests for the shared contracts: the checker itself, the example files, and the node registry."""

import json

import pandas as pd
import pytest

from common.contracts import load_node_ids, load_schema, validate_csv, validate_dataframe
from common.paths import CONTRACTS_DIR, EDGES_CSV, FIXTURES_DIR, NODES_CSV, SCHEMAS_DIR

SCHEMA_NAMES = sorted(p.name.replace(".schema.json", "") for p in SCHEMAS_DIR.glob("*.schema.json"))


# ---------- every example file must satisfy its own schema ----------

@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_fixture_exists_and_passes(name):
    """Each schema ships with a tiny valid example file named after it."""
    path = FIXTURES_DIR / f"{name}.csv"
    assert path.exists(), f"missing example file for schema '{name}'"
    report = validate_csv(name, path)
    assert report.ok, [str(v) for v in report.violations]


# ---------- the checker must actually catch mistakes ----------

def _fixture(name):
    return pd.read_csv(FIXTURES_DIR / f"{name}.csv", dtype=str)


def _rules(report):
    return {v.rule for v in report.violations}


def test_detects_missing_required_column():
    df = _fixture("telemetry").drop(columns=["dwell_time_hours"])
    assert "missing_column" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_optional_column_may_be_absent():
    df = _fixture("hourly_node_telemetry").drop(columns=["vessels_waiting"])
    assert validate_dataframe(df, load_schema("hourly_node_telemetry")).ok


def test_detects_out_of_range_value():
    df = _fixture("telemetry")
    df.loc[0, "weather_severity_index"] = "11.5"
    assert "above_max_10" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_utilization_above_100_is_allowed():
    """Overload is a real signal, so the contract must not treat >100% as an error."""
    df = _fixture("telemetry")
    df.loc[0, "capacity_utilization_pct"] = "118.0"
    assert validate_dataframe(df, load_schema("telemetry")).ok


def test_detects_value_not_in_allowed_list():
    df = _fixture("telemetry")
    df.loc[0, "delay_status"] = "LATE"
    assert "value_not_allowed" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_detects_unknown_node_id():
    df = _fixture("telemetry")
    df.loc[0, "current_node_id"] = "PORT_XX_NOPE"
    assert "unknown_node_id" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_detects_duplicate_primary_key():
    df = pd.concat([_fixture("telemetry"), _fixture("telemetry").iloc[[0]]], ignore_index=True)
    assert "duplicate_primary_key" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_detects_null_in_required_value():
    df = _fixture("telemetry")
    df.loc[0, "dwell_time_hours"] = None
    assert "null_not_allowed" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_nullable_column_accepts_null():
    df = _fixture("incident_messages")
    df.loc[0, "urgency_label"] = None
    assert validate_dataframe(df, load_schema("incident_messages")).ok


def test_detects_bad_timestamp():
    df = _fixture("telemetry")
    df.loc[0, "timestamp"] = "05/01/2026 08:10"
    assert "not_iso_datetime" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_detects_probabilities_not_summing_to_one():
    df = _fixture("ml_predictions")
    df.loc[0, "p_critical"] = "0.10"
    assert "probabilities_do_not_sum_to_1" in _rules(validate_dataframe(df, load_schema("ml_predictions")))


def test_detects_pattern_mismatch():
    df = _fixture("telemetry")
    df.loc[0, "scenario_id"] = "scenario-7"
    assert "pattern_mismatch" in _rules(validate_dataframe(df, load_schema("telemetry")))


def test_detects_non_integer_in_int_column():
    df = _fixture("dl_forecasts")
    df.loc[0, "horizon_h"] = "1.5"
    assert "not_int" in _rules(validate_dataframe(df, load_schema("dl_forecasts")))


def test_extra_columns_are_reported_but_allowed():
    df = _fixture("telemetry")
    df["my_new_feature"] = "1"
    report = validate_dataframe(df, load_schema("telemetry"))
    assert report.ok and report.extra_columns == ["my_new_feature"]


def test_extends_merges_parent_columns():
    columns = {c["name"] for c in load_schema("scenario_telemetry")["columns"]}
    assert {"hour_ts", "node_id", "severity", "is_disrupted_hour"} <= columns


def test_unknown_schema_name_gives_helpful_error():
    with pytest.raises(FileNotFoundError) as err:
        load_schema("does_not_exist")
    assert "Known schemas" in str(err.value)


# ---------- node registry integrity ----------

def test_node_registry_is_consistent():
    nodes = pd.read_csv(NODES_CSV, keep_default_na=False)
    assert nodes["node_id"].is_unique
    assert set(nodes["node_type"]) <= {"PORT", "RAIL_HUB", "DC", "CHOKEPOINT"}
    assert (nodes["aliases"].str.len() > 0).all(), "every node needs at least one alias for the NER linker"
    assert nodes["lat"].between(-90, 90).all() and nodes["lon"].between(-180, 180).all()


def test_edges_only_use_known_nodes():
    nodes = load_node_ids()
    edges = pd.read_csv(EDGES_CSV)
    assert edges["edge_id"].is_unique
    assert set(edges["src"]) <= nodes and set(edges["dst"]) <= nodes
    assert (edges["src"] != edges["dst"]).all()
    assert set(edges["mode"]) <= {"SEA", "RAIL", "TRUCK", "BARGE", "ADJACENT"}


def test_every_node_is_connected():
    """An isolated node would be useless for the graph model."""
    nodes = load_node_ids()
    edges = pd.read_csv(EDGES_CSV)
    connected = set(edges["src"]) | set(edges["dst"])
    assert nodes == connected


def test_glossary_targets_exist():
    glossary = pd.read_csv(CONTRACTS_DIR / "jargon_glossary.csv", keep_default_na=False)
    nodes = load_node_ids()
    for targets in glossary["maps_to"]:
        for node_id in filter(None, targets.split("|")):
            assert node_id in nodes, f"glossary points to unknown node {node_id}"
    assert glossary["term"].is_unique


# ---------- policy files ----------

def test_policy_files_are_valid_json_and_consistent():
    label = json.loads((CONTRACTS_DIR / "label_policy.json").read_text(encoding="utf-8"))
    cost = json.loads((CONTRACTS_DIR / "cost_policy.json").read_text(encoding="utf-8"))
    feature = json.loads((CONTRACTS_DIR / "feature_policy.json").read_text(encoding="utf-8"))

    classes = label["classes"]
    matrix = cost["cost_matrix_rows_true_cols_pred"]
    assert matrix["classes"] == classes
    for true_class in classes:
        assert len(matrix[true_class]) == len(classes)
    # Correct predictions must cost nothing.
    for i, true_class in enumerate(classes):
        assert matrix[true_class][i] == 0.0

    # The label basis and the target must never be usable as features.
    assert {label["basis_column"], label["target_column"]} <= set(feature["forbidden_features"])
    assert not set(feature["allowed_base_features"]) & set(feature["forbidden_features"])


# ---------- write_validated_csv: the safe way to write a hand-over file ----------

def test_write_validated_csv_accepts_typed_dataframe(tmp_path):
    """Role scripts hold real dtypes (floats, bools, Timestamps), not text, so the checker must cope."""
    from common.contracts import write_validated_csv

    df = pd.read_csv(FIXTURES_DIR / "telemetry.csv", parse_dates=["timestamp"])
    out = tmp_path / "prepared.csv"
    write_validated_csv(df, "telemetry", out)
    assert out.exists() and validate_csv("telemetry", out).ok


def test_write_validated_csv_refuses_broken_data(tmp_path):
    from common.contracts import write_validated_csv

    df = pd.read_csv(FIXTURES_DIR / "telemetry.csv")
    df.loc[0, "labor_deficit_ratio"] = 3.0  # ratio must be between 0 and 1
    out = tmp_path / "bad.csv"
    with pytest.raises(ValueError) as err:
        write_validated_csv(df, "telemetry", out)
    assert "labor_deficit_ratio" in str(err.value)
    assert not out.exists()
