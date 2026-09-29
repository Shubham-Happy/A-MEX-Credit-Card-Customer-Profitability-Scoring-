from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import percentile_rank, safe_qcut, zscore

F_COLS = [f"f{i}" for i in range(1, 24)]


def _clip_nonnegative(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].clip(lower=0)


def _segment_median_estimate(
    df: pd.DataFrame,
    target: str,
    observed_mask: pd.Series,
    group_cols: list[str],
    rho: float = 0.55,
    k: float = 500.0,
) -> pd.Series:
    """Empirical-Bayes segment-median estimate.

    Estimate target for all rows using observed rows grouped by group_cols.
    Segment medians are shrunk toward the global median based on segment count.
    """
    global_median = pd.to_numeric(df.loc[observed_mask, target], errors="coerce").median()
    if np.isnan(global_median):
        global_median = 0.0

    stats = (
        df.loc[observed_mask, group_cols + [target]]
        .groupby(group_cols, dropna=False)[target]
        .agg(["median", "count"])
        .reset_index()
    )
    stats["lambda"] = rho * stats["count"] / (stats["count"] + k)
    stats[f"{target}_eb"] = stats["lambda"] * stats["median"] + (1 - stats["lambda"]) * global_median
    merged = df[group_cols].merge(stats[group_cols + [f"{target}_eb"]], how="left", on=group_cols)
    return merged[f"{target}_eb"].fillna(global_median).astype(float)


def make_base_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Common feature base with decoy repairs and conservative imputations.

    This produces every term needed by baseline_086, attempt7_refined, and
    attempt8_plus_travelpremium.
    """
    df = df_raw.copy()
    for c in ["id"] + F_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")

    # Keep ID outside all computation.
    out = pd.DataFrame({"ID": df["id"].astype(int)})

    # Numeric conversion.
    for c in F_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Preserve masks for documentation/diagnostics.
    for c in F_COLS:
        out[f"{c}_missing"] = df[c].isna().astype(int)

    # Core fills.
    f1_imp = df["f1"].fillna(0).clip(lower=0)
    f2_imp = df["f2"].fillna(0).clip(lower=0)
    f3_imp = df["f3"].fillna(0).clip(lower=0)
    f13_zero = df["f13"].fillna(0).clip(lower=0)
    f21_imp = df["f21"].fillna(0).clip(lower=0)
    f17_imp = df["f17"].fillna(0).clip(lower=0)
    f18_imp = df["f18"].fillna(0).clip(lower=0)

    # Conservative risk imputation: median first, then optional EB push-up.
    f11_med = df["f11"].median()
    f11_temp = df["f11"].fillna(f11_med)
    risk_pct_temp = percentile_rank(f11_temp)

    # Initial decoy repairs.
    cancel_custom = np.maximum(f2_imp, f3_imp)
    line_custom = np.maximum(f17_imp, f18_imp)

    # Category observed block and raw category sum.
    cat_cols = ["f6", "f7", "f8", "f9", "f10"]
    cat_obs = df[cat_cols].notna().all(axis=1)
    cat_sum_zero = df[cat_cols].fillna(0).sum(axis=1)

    # Segment definitions for EB imputations.
    seg = pd.DataFrame(index=df.index)
    seg["risk_band"] = safe_qcut(f11_temp, q=10)
    seg["revolve_band"] = safe_qcut(f1_imp, q=10)
    seg["cancel_flag"] = (cancel_custom > 0).astype(int)
    seg["collection_flag"] = (f3_imp > 0).astype(int)
    seg["line_present"] = ((df["f17"].notna()) | (df["f18"].notna())).astype(int)
    seg["reward_present"] = df["f21"].notna().astype(int)
    seg["lounge_flag"] = (f13_zero > 0).astype(int)
    seg["category_observed"] = cat_obs.astype(int)

    work = pd.concat([df, seg], axis=1)
    group_cols_cat = ["risk_band", "revolve_band", "cancel_flag", "collection_flag", "line_present", "lounge_flag"]

    # EB category spend imputation: observed category sum trusted, missing rows get capped gated estimate.
    work["CatSum"] = cat_sum_zero.where(cat_obs, np.nan)
    spend_eb = _segment_median_estimate(work, "CatSum", cat_obs, group_cols_cat, rho=0.50, k=500)
    p40_cat_sum = float(work.loc[cat_obs, "CatSum"].quantile(0.40)) if cat_obs.any() else 0.0
    alpha = 0.30 * (1 - risk_pct_temp) * (1 - seg["collection_flag"])
    spend_missing_est = np.minimum(alpha * spend_eb, p40_cat_sum)
    spend_custom_imp = cat_sum_zero.where(cat_obs, spend_missing_est).clip(lower=0)

    # Individual category imputations for TravelPremium. Conservative EB with feature-specific P40 caps.
    cat_imp = {}
    for col in cat_cols:
        observed = df[col].notna()
        work[col + "_obs_target"] = df[col].where(observed, np.nan)
        eb = _segment_median_estimate(work.rename(columns={col + "_obs_target": "target_tmp"}), "target_tmp", observed, group_cols_cat, rho=0.45, k=500)
        cap = float(df.loc[observed, col].quantile(0.40)) if observed.any() else 0.0
        est = np.minimum(alpha * eb, cap)
        cat_imp[col] = df[col].where(observed, est).fillna(0).clip(lower=0)

    # Risk EB: missing risk never below median. Grouped on safe distress/exposure indicators.
    group_cols_risk = ["cancel_flag", "collection_flag", "revolve_band", "line_present", "category_observed"]
    work["f11_target"] = df["f11"]
    f11_eb = _segment_median_estimate(work.rename(columns={"f11_target": "target_tmp"}), "target_tmp", df["f11"].notna(), group_cols_risk, rho=0.50, k=500)
    f11_imp = df["f11"].copy()
    missing_risk = f11_imp.isna()
    f11_imp.loc[missing_risk] = np.maximum(f11_med, 0.5 * f11_med + 0.5 * f11_eb.loc[missing_risk])
    risk_pct = percentile_rank(f11_imp)

    # Lounge EB: weak/capped imputation, because it is a low-weight cost term.
    work["spend_band"] = safe_qcut(spend_custom_imp, q=10)
    group_cols_lounge = ["spend_band", "risk_band", "reward_present", "line_present"]
    work["f13_target"] = df["f13"]
    lounge_eb = _segment_median_estimate(work.rename(columns={"f13_target": "target_tmp"}), "target_tmp", df["f13"].notna(), group_cols_lounge, rho=0.35, k=500)
    lounge_cap = float(df.loc[df["f13"].notna(), "f13"].quantile(0.75)) if df["f13"].notna().any() else 0.0
    f13_imp = f13_zero.copy()
    missing_lounge = df["f13"].isna()
    f13_imp.loc[missing_lounge] = np.minimum(0.5 * lounge_eb.loc[missing_lounge], lounge_cap)

    # Risk-adjusted lend quality: capacity only helps when risk is controlled.
    lend_quality = line_custom * (1 - 0.65 * risk_pct)
    lend_quality = lend_quality.clip(lower=0)

    # Export raw/imputed terms.
    out["f1_imp"] = f1_imp
    out["f11_imp"] = f11_imp
    out["f13_imp"] = f13_imp
    out["f21_imp"] = f21_imp
    out["Cancel_custom"] = cancel_custom
    out["Line_custom"] = line_custom
    out["RiskPct"] = risk_pct
    out["Spend_custom_zero"] = cat_sum_zero.clip(lower=0)
    out["Spend_custom_imp"] = spend_custom_imp
    out["LendQuality_imp"] = lend_quality
    for col in cat_cols:
        out[col + "_imp"] = cat_imp[col]

    # Standardized terms.
    out["z_revolve"] = zscore(out["f1_imp"])
    out["z_spend_zero"] = zscore(out["Spend_custom_zero"])
    out["z_spend_imp"] = zscore(out["Spend_custom_imp"])
    out["z_risk"] = zscore(out["f11_imp"])
    out["z_lounge"] = zscore(out["f13_imp"])
    out["z_reward"] = zscore(out["f21_imp"])
    out["z_cancel"] = zscore(out["Cancel_custom"])
    out["z_lend"] = zscore(out["LendQuality_imp"])

    z_f6 = zscore(out["f6_imp"])
    z_f9 = zscore(out["f9_imp"])
    z_f10 = zscore(out["f10_imp"])
    out["TravelPremium_imp"] = 0.45 * z_f6 + 0.35 * z_f9 + 0.20 * z_f10
    out["z_travel_premium"] = zscore(out["TravelPremium_imp"])

    return out
