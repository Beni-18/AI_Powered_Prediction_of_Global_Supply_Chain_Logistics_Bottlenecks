"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Loads slices of data based on the immutable chronological split column.

Owner: Role 1 - ML Engineer
"""

import pandas as pd


def load_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return an isolated copy of rows matching the requested split ('train', 'val', or 'test').

    Guarantees no leakage between partitions and validates input integrity.
    """
    # Verify that the split column exists so we do not guess partitions.
    if "split" not in df.columns:
        raise KeyError("Dataframe is missing the required 'split' column.")

    valid_splits = {"train", "val", "test"}
    normalized = split_name.lower().strip()
    if normalized not in valid_splits:
        raise ValueError(f"Split '{split_name}' is invalid. Must be one of {sorted(valid_splits)}.")

    # Filter rows strictly belonging to the requested partition.
    mask = df["split"] == normalized
    subset = df[mask].copy()

    # Ensure the subset is not empty before handing it to a model.
    if subset.empty:
        raise ValueError(f"No records found for split='{normalized}'.")

    return subset
