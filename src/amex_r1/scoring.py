from __future__ import annotations

import pandas as pd

from .utils import minmax


def score_variant(features: pd.DataFrame, variant: str = "attempt8_plus_travelpremium") -> pd.DataFrame:
    """Return ID, RawScore, Prediction for a named scorecard variant."""
    if variant == "baseline_086":
        raw = (
            0.35 * features["z_revolve"]
            + 0.25 * features["z_spend_zero"]
            - 0.30 * features["z_risk"]
            - 0.05 * features["z_lounge"]
            - 0.10 * features["z_reward"]
            - 0.15 * features["z_cancel"]
        )
    elif variant == "attempt7_refined_eb":
        raw = (
            0.34 * features["z_revolve"]
            + 0.24 * features["z_spend_imp"]
            + 0.04 * features["z_lend"]
            - 0.31 * features["z_risk"]
            - 0.05 * features["z_lounge"]
            - 0.10 * features["z_reward"]
            - 0.15 * features["z_cancel"]
        )
    elif variant == "attempt8_rebalanced":
        raw = (
            0.36 * features["z_revolve"]
            + 0.26 * features["z_spend_imp"]
            + 0.02 * features["z_lend"]
            - 0.29 * features["z_risk"]
            - 0.05 * features["z_lounge"]
            - 0.10 * features["z_reward"]
            - 0.14 * features["z_cancel"]
        )
    elif variant == "attempt8_plus_travelpremium":
        raw = (
            0.36 * features["z_revolve"]
            + 0.26 * features["z_spend_imp"]
            + 0.02 * features["z_lend"]
            + 0.03 * features["z_travel_premium"]
            - 0.29 * features["z_risk"]
            - 0.05 * features["z_lounge"]
            - 0.10 * features["z_reward"]
            - 0.14 * features["z_cancel"]
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    pred = minmax(raw)
    return pd.DataFrame({"ID": features["ID"].astype(int), "RawScore": raw, "Prediction": pred})


def framework_text(variant: str) -> dict[str, str]:
    if variant == "attempt8_plus_travelpremium":
        equation = (
            "Raw_A8P = 0.36*z(f1_imp) + 0.26*z(Spend_custom_imp) "
            "+ 0.02*z(LendQuality_imp) + 0.03*z(TravelPremium_imp) "
            "- 0.29*z(f11_imp) - 0.05*z(f13_imp) - 0.10*z(f21_imp) "
            "- 0.14*z(Cancel_custom). Prediction = min-max scaled Raw_A8P."
        )
        variables = (
            "Revenue: f1 average revolve balance, reconstructed category spend f6-f10. "
            "Premier-fit: f6 airline/travel, f9 lodging, f10 dining through TravelPremium. "
            "Capacity: repaired f17/f18 lending line through risk-adjusted LendQuality. "
            "Risk/cost: f11 risk, f13 lounge access, f21 rewards redeemed, f2/f3 cancellation/collection calls."
        )
        derivation = (
            "Weights are a low-radius rebalancing of the 0.86 public-score P&L anchor. "
            "Revolve and reconstructed spend receive the largest positive weights because they proxy interest and interchange revenue. "
            "Risk remains the dominant negative term because credit losses can dominate issuer economics. "
            "Rewards, lounge usage, and cancellation are cost/future-value leakage terms. "
            "LendQuality and TravelPremium are deliberately small positive refinements to avoid public-LB overfitting."
        )
    else:
        equation = "See scoring.py for the selected variant formula."
        variables = "See scoring.py and features.py."
        derivation = "Variant-specific scorecard in scoring.py."

    return {
        "Variables Used": variables,
        "Profitability Equation": equation,
        "Prediction Logic": (
            "Each cardmember receives one raw profitability score from a transparent card-issuer P&L equation. "
            "Revenue-like signals increase the score, while risk, benefit usage, reward redemption, and cancellation/collection signals reduce it. "
            "The final min-max scaling is applied only after scoring and does not change rank."
        ),
        "Variable Selection Logic": (
            "The selected variables map directly to Premier Card issuer economics: revolving balance for interest income, category spend for interchange, "
            "risk score for expected credit loss, rewards and lounge usage for issuer costs, and cancellation/collection calls for churn/distress. "
            "Raw f5 was excluded after aggregate-consistency checks showed it did not match category spend totals."
        ),
        "Coefficient/Weight Derivation": derivation,
        "Feature Transformations": (
            "Aggregate correction: Spend_custom = f6+f7+f8+f9+f10; Cancel_custom = max(f2,f3); Line_custom = max(f17,f18). "
            "Missing values are handled with selective conservative imputation: missing category spend receives Empirical-Bayes segment estimates that are risk-gated and capped; "
            "missing risk is never imputed below median; missing lounge usage receives a capped partial segment estimate; missing rewards are treated as no observed redemption. "
            "All score terms are z-standardized before weighting."
        ),
        "Business Logic": (
            "Profitability is approximated as revenue minus expected loss and servicing/benefit costs. "
            "Premier travel affinity is included as a small positive signal because the product is positioned around airline, lodging, dining, and travel/lifestyle value. "
            "Lending capacity is included only after risk adjustment."
        ),
        "Assumptions": (
            "Missing activity does not always equal zero, so spend and lounge missingness are treated cautiously with shrinkage rather than optimistic full imputation. "
            "A missing risk score is not interpreted as risk-free. Raw f5 is treated as an unreliable aggregate/decoy and is excluded from the final score."
        ),
        "Validation Approach": (
            "Validation is no-label and rank-focused: row/ID integrity checks, null checks, comparison against the 0.86 public-score workbook, top-20 overlap, "
            "rank correlations, and top-20 profile diagnostics for revenue, risk, cost, churn, lending capacity, and Premier travel affinity."
        ),
        "Additional Notes": (
            "The method intentionally stays explainable and low-radius because the final score is evaluated on public/private top-20 overlap. "
            "Heavy black-box imputers were avoided to preserve auditability and reduce public-LB overfitting risk."
        ),
    }
