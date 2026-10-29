"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Guards against data leakage and forbidden feature usage by enforcing contracts/feature_policy.json.

Owner: Role 1 - ML Engineer
Reads: contracts/feature_policy.json
"""

import json
from pathlib import Path
from typing import Any, Iterable

from common.paths import REPO_ROOT

POLICY_PATH = REPO_ROOT / "contracts" / "feature_policy.json"


def get_feature_policy() -> dict[str, Any]:
    """Load and return the frozen feature policy specification as a dictionary."""
    # Open the frozen contract JSON file that defines which features are legal.
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_features_allowed(cols: Iterable[str]) -> None:
    """Ensure that all feature columns conform to the zero-leakage contract.

    Raises ValueError if:
    1. Any forbidden feature (like realized_delay_hours, delay_status, IDs) is present.
    2. Any feature is not explicitly in allowed_base_features and does not use an allowed derived prefix.
    """
    policy = get_feature_policy()
    forbidden = set(policy.get("forbidden_features", []))
    allowed_base = set(policy.get("allowed_base_features", []))
    allowed_prefixes = tuple(policy.get("allowed_derived_prefixes", []))

    columns = list(cols)

    # Step 1: Check for forbidden columns that cause target or bookkeeping leakage.
    leaked_cols = [col for col in columns if col in forbidden]
    if leaked_cols:
        raise ValueError(
            f"Zero-leakage policy violation: Found forbidden columns {leaked_cols}. "
            f"Reason: {policy.get('forbidden_reason', 'Leakage hazard')}."
        )

    # Step 2: Ensure every feature is either an allowed base feature or starts with an allowed derived prefix.
    disallowed_cols = []
    for col in columns:
        is_base = col in allowed_base
        is_derived = col.startswith(allowed_prefixes)
        if not (is_base or is_derived):
            disallowed_cols.append(col)

    if disallowed_cols:
        raise ValueError(
            f"Unapproved feature violation: Columns {disallowed_cols} are not in allowed_base_features "
            f"and do not start with allowed prefixes {allowed_prefixes}."
        )
