````markdown
# AMEX Campus Challenge 2026 Repository

The goal of Round 1 is to assign a profitability score to every cardmember using only the provided variables `f1`–`f23`, then rank all customers so that the predicted **top 20% most profitable cardmembers** overlaps as much as possible with the hidden ground-truth top 20%.

---

## Layout

```text
A-MEX/
├─ src/
│  └─ amex_r1/
│     ├─ __init__.py
│     ├─ build_submission.py      # CLI entrypoint
│     ├─ features.py              # decoy correction + conservative imputations
│     ├─ scoring.py               # scorecard equations
│     ├─ io.py                    # CSV/XLSX I/O + framework writing
│     ├─ validate.py              # mechanical submission checks
│     ├─ compare_submissions.py   # optional comparison against baseline workbook
│     └─ utils.py                 # z-score, min-max, percentile rank, qcut helpers
├─ data/
│  └─ raw/
│     └─ .gitkeep
├─ outputs/
│  └─ .gitkeep
├─ README.md
├─ requirements.txt
├─ pyproject.toml
└─ .gitignore
````

---

## Setup

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

---

## Required input files

Place the competition files inside:

```text
data/raw/
```

Expected files:

```text
data/raw/R1dataset.csv
data/raw/submission_template.xlsx
```

Optional, for documentation/reference only:

```text
data/raw/feature_description.csv
```

The raw dataset is expected to contain:

```text
id, f1, f2, ..., f23
```

The `id` column is used only for output alignment. It is never used as a scoring feature.

---

## Build submission

Linux/macOS:

```bash
python -m amex_r1.build_submission \
  --data data/raw/R1dataset.csv \
  --template data/raw/submission_template.xlsx \
  --output-xlsx outputs/amex_attempt8_plus_travelpremium.xlsx \
  --output-csv outputs/amex_attempt8_plus_travelpremium_predictions.csv \
  --variant attempt8_plus_travelpremium
```

Windows PowerShell:

```powershell
python -m amex_r1.build_submission `
  --data data/raw/R1dataset.csv `
  --template data/raw/submission_template.xlsx `
  --output-xlsx outputs/amex_attempt8_plus_travelpremium.xlsx `
  --output-csv outputs/amex_attempt8_plus_travelpremium_predictions.csv `
  --variant attempt8_plus_travelpremium
```

The command produces:

```text
outputs/amex_attempt8_plus_travelpremium.xlsx
outputs/amex_attempt8_plus_travelpremium_predictions.csv
```

The XLSX file is the primary submission artifact.

---

# Method overview

The workflow is a business-first profitability scorecard.

It follows the issuer P&L structure:

```text
Profitability ≈ Revenue − Credit Risk − Rewards Cost − Benefit Cost − Churn / Distress
```

Positive signals:

```text
revolving balance
total reconstructed spend
risk-adjusted lend-line capacity
premium travel-category spend
```

Negative signals:

```text
risk score
rewards redeemed
lounge usage
cancellation / collection-related cancellation behavior
```

The method is intentionally not a black-box supervised model because the dataset does not contain a target profitability label. The objective is to create a scalable and explainable profitability ranking from the provided attributes.

---

# Workflow

```text
Raw R1dataset.csv
↓
Schema validation
↓
Missing-value handling
↓
Aggregate decoy correction
↓
Empirical-Bayes-style conservative imputation
↓
Feature standardization
↓
Scorecard equation
↓
Final min-max scaling
↓
Submission XLSX generation
↓
Mechanical validation
```

---

# 1. Schema validation

The pipeline first checks that:

```text
- all expected columns are present
- id is present
- id is unique
- no row is added, removed, or reordered
- all 500,000 rows are scored
```

The `id` column is separated immediately and is carried only for final submission writing.

---

# 2. Decoy correction

Exploratory analysis showed that some aggregate columns violate logical relationships with their component columns.

The workflow therefore reconstructs or repairs these aggregate signals before scoring.

## 2.1 Spend correction

Raw `f5` is excluded from the primary score because it does not behave like a valid total-spend field.

Instead:

```text
Spend_custom = f6 + f7 + f8 + f9 + f10
```

This reconstructed spend signal is used as the main interchange/spend-revenue proxy.

## 2.2 Cancellation correction

Collection-related cancellation calls should not logically exceed total cancellation calls. To repair this relationship:

```text
Cancel_custom = max(f2, f3)
```

This creates a stronger churn/distress signal.

## 2.3 Lending-line correction

Consumer lend line should not logically exceed total lend line. To repair this relationship:

```text
Line_custom = max(f17, f18)
```

This repaired lend-line signal is not used directly as pure revenue. It is converted into a risk-adjusted capacity signal.

---

# 3. Missing-value strategy

Missing values are not handled with a single global rule.

The repository uses a conservative, feature-specific policy:

```text
activity missingness → no observed activity or small capped inference
risk missingness     → conservative imputation, never artificially safe
lending missingness  → logical repair if one field is present, zero if both are absent
reward missingness   → no observed redemption
```

The guiding principle is:

```text
Observed value > reliable inferred value > neutral/zero fallback > unsafe fake precision
```

---

## 3.1 Category spend imputation

The category spend block `f6`–`f10` is important because raw `f5` is not trusted.

When category spends are present:

```text
Spend_custom_imp = f6 + f7 + f8 + f9 + f10
```

When the category block is missing, the pipeline applies conservative segment-based imputation.

The imputation is intentionally capped so missing rows cannot become artificial high-spend “whales.”

Conceptually:

```text
Spend_EB = segment_median(Spend_custom | similar risk / revolve / cancellation / line / lounge profile)

Spend_custom_imp =
    observed category sum, if f6–f10 are present
    conservative capped Spend_EB, if f6–f10 are missing
```

The imputation is reduced for risky or collection-heavy customers.

---

## 3.2 Risk imputation

Missing `f11` values are not filled with zero.

A zero risk score would incorrectly imply that the customer is perfectly safe.

Instead:

```text
f11_imp = f11, if observed
f11_imp = conservative segment-based value, if missing
```

The imputed risk value is constrained so that missing risk is never treated as better than the population median.

---

## 3.3 Lounge imputation

Lounge usage `f13` is a cost signal.

For observed values:

```text
f13_imp = f13
```

For missing values, the pipeline applies a small, capped segment-level imputation. This avoids over-penalizing customers with missing lounge records.

---

## 3.4 Rewards redeemed imputation

Rewards redeemed `f21` is treated conservatively.

```text
f21_imp = f21, if observed
f21_imp = 0, if missing
```

The model does not invent synthetic rewards redemption because that would introduce an uncertain cost penalty.

---

## 3.5 Lending-line imputation

For `f17` and `f18`:

```text
if either f17 or f18 is observed:
    Line_custom = max(f17, f18)

if both are missing:
    Line_custom = 0
```

No positive lend-line value is imputed when both lending fields are missing.

---

# 4. Engineered features

## 4.1 Risk percentile

The risk percentile is computed from the imputed risk score:

```text
RiskPct = percentile_rank(f11_imp)
```

Higher `RiskPct` means higher estimated risk.

---

## 4.2 Risk-adjusted lend quality

Lending capacity is valuable only when the cardmember is not too risky.

```text
LendQuality_imp = Line_custom × (1 − 0.65 × RiskPct)
```

This allows high lend-line customers to receive a positive contribution only when their risk profile is acceptable.

---

## 4.3 TravelPremium feature

The Premier Card proposition is strongly tied to travel and lifestyle spend.

The workflow therefore adds a small premium-category signal from airline, lodging, and dining spend:

```text
TravelPremium =
    0.45 × z(f6_imp)
  + 0.35 × z(f9_imp)
  + 0.20 × z(f10_imp)
```

This feature is deliberately low-weighted because premium-category spend can also generate higher rewards cost.

---

# 5. Standardization

Before combining terms, each scoring feature is standardized using z-score scaling:

```text
z(x) = (x − mean(x)) / std(x)
```

This prevents large-scale financial variables from dominating smaller count-based variables only because of units.

The final score is then min-max scaled:

```text
Prediction = minmax(RawScore)
```

The final min-max scaling is rank-preserving. It does not change the ordering of customers.

---

# 6. Current primary scorecard

Current primary variant:

```text
variant = attempt8_plus_travelpremium
```

Raw score:

```text
RawScore =
    0.36 × z(f1_imp)
  + 0.26 × z(Spend_custom_imp)
  + 0.02 × z(LendQuality_imp)
  + 0.03 × z(TravelPremium)
  − 0.29 × z(f11_imp)
  − 0.05 × z(f13_imp)
  − 0.10 × z(f21_imp)
  − 0.14 × z(Cancel_custom)
```

Final prediction:

```text
Prediction = minmax(RawScore)
```

---

# 7. Business interpretation of weights

| Term               |  Weight | Interpretation                                                                                   |
| ------------------ | ------: | ------------------------------------------------------------------------------------------------ |
| `f1_imp`           | `+0.36` | Revolving balance is treated as the strongest revenue signal because it proxies interest income. |
| `Spend_custom_imp` | `+0.26` | Reconstructed total spend proxies interchange revenue and card activity.                         |
| `LendQuality_imp`  | `+0.02` | Repaired lend-line capacity is useful only after risk adjustment, so the weight is small.        |
| `TravelPremium`    | `+0.03` | Premium travel/lodging/dining spend indicates Premier Card fit.                                  |
| `f11_imp`          | `−0.29` | Higher risk score reduces expected profitability due to potential credit losses.                 |
| `f13_imp`          | `−0.05` | Lounge usage is treated as a direct cardmember-service cost.                                     |
| `f21_imp`          | `−0.10` | Rewards redeemed are treated as realized rewards liability.                                      |
| `Cancel_custom`    | `−0.14` | Cancellation and collection-related calls reduce expected future value.                          |

---

# 8. Validation

The submission builder performs mechanical validation before writing the final output.

Checks include:

```text
- prediction row count matches input row count
- every input ID appears exactly once
- no duplicate IDs
- no missing predictions
- prediction values are numeric
- id is not used in score computation
- output workbook contains the required Predictions sheet
```

Optional comparison can be performed against a previous workbook using:

```bash
python -m amex_r1.compare_submissions \
  --candidate outputs/amex_attempt8_plus_travelpremium.xlsx \
  --baseline path/to/baseline.xlsx \
  --output outputs/comparison.csv
```

The comparison utility is useful for checking:

```text
- rank correlation
- top-20% overlap
- top-10% overlap
- top-5% overlap
- customers entering/leaving the predicted top 20%
```

---

# 9. Safety notes

* `id` is never used as a predictor.
* No rows are added, removed, or reordered.
* Raw `f5` is excluded because it failed aggregate consistency checks against `f6–f10`.
* Missing risk is conservatively imputed and never allowed to become artificially safe.
* Missing spend is imputed only conservatively and is capped.
* Missing reward redemption is not artificially inferred.
* Lend-line capacity is risk-adjusted before contributing positively.
* Final min-max scaling is rank-preserving.
* The method is deliberately explainable and spreadsheet-auditable.

---

# 10. Reproducibility

The workflow is deterministic.

Given the same:

```text
R1dataset.csv
submission_template.xlsx
variant name
```

The pipeline will produce the same:

```text
Predictions
Profitability Framework
```
