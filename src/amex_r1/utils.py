from __future__ import annotations

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def zscore(s: pd.Series) -> pd.Series:
    """Population z-score used by sklearn StandardScaler."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mean = x.mean()
    std = x.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mean) / std


def minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = x.min(), x.max()
    if hi == lo or np.isnan(hi) or np.isnan(lo):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - lo) / (hi - lo)


def percentile_rank(s: pd.Series) -> pd.Series:
    """Dense-ish percentile rank in [0, 1]. Used for risk gating."""
    return pd.to_numeric(s, errors="coerce").rank(method="average", pct=True).fillna(0.5)


def safe_qcut(s: pd.Series, q: int, labels: bool = False) -> pd.Series:
    """qcut with duplicate-edge fallback."""
    x = pd.to_numeric(s, errors="coerce").fillna(pd.to_numeric(s, errors="coerce").median())
    try:
        return pd.qcut(x, q=q, labels=labels, duplicates="drop")
    except ValueError:
        # All values identical or too few unique values; one neutral bin.
        return pd.Series(np.zeros(len(x), dtype=int), index=s.index)


def signed_log1p(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return np.sign(x) * np.log1p(np.abs(x))
