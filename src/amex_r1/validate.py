from __future__ import annotations

import pandas as pd


def validate_predictions(raw_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    if len(pred_df) != len(raw_df):
        raise ValueError(f"Prediction row count mismatch: {len(pred_df)} vs {len(raw_df)}")
    if pred_df["ID"].duplicated().any():
        raise ValueError("Duplicate IDs in predictions")
    if set(pred_df["ID"].astype(int)) != set(raw_df["id"].astype(int)):
        raise ValueError("Prediction ID set does not match raw data ID set")
    if pred_df["Prediction"].isna().any():
        raise ValueError("NaN predictions detected")
    if not pd.api.types.is_numeric_dtype(pred_df["Prediction"]):
        raise ValueError("Prediction column must be numeric")
