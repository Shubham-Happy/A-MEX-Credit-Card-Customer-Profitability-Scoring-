from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io import load_predictions_xlsx


def compare_predictions(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float | int]:
    cand = candidate.rename(columns={"Prediction": "Prediction_candidate"})
    base = baseline.rename(columns={"Prediction": "Prediction_baseline"})
    merged = base.merge(cand, on="ID", how="inner")
    if len(merged) != len(base) or len(merged) != len(cand):
        raise ValueError("Candidate and baseline IDs do not match exactly")

    out: dict[str, float | int] = {}
    out["n_rows"] = int(len(merged))
    out["pearson"] = float(merged["Prediction_baseline"].corr(merged["Prediction_candidate"], method="pearson"))
    out["spearman"] = float(merged["Prediction_baseline"].corr(merged["Prediction_candidate"], method="spearman"))

    n = len(merged)
    for frac in [0.20, 0.10, 0.05, 0.01]:
        k = int(round(n * frac))
        top_base = set(merged.nlargest(k, "Prediction_baseline")["ID"])
        top_cand = set(merged.nlargest(k, "Prediction_candidate")["ID"])
        overlap = len(top_base & top_cand)
        tag = f"top_{int(frac * 100)}"
        out[f"{tag}_k"] = k
        out[f"{tag}_overlap"] = overlap
        out[f"{tag}_overlap_pct"] = overlap / k
    out["entered_top20"] = out["top_20_k"] - out["top_20_overlap"]
    out["left_top20"] = out["top_20_k"] - out["top_20_overlap"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate submission against baseline workbook.")
    parser.add_argument("--candidate", required=True, help="Candidate XLSX or CSV")
    parser.add_argument("--baseline", required=True, help="Baseline XLSX or CSV")
    parser.add_argument("--output", required=True, help="Output comparison CSV")
    args = parser.parse_args()

    def read(path: str) -> pd.DataFrame:
        if path.lower().endswith(".xlsx"):
            return load_predictions_xlsx(path)
        return pd.read_csv(path)

    cand = read(args.candidate)
    base = read(args.baseline)
    metrics = compare_predictions(cand, base)
    out = pd.DataFrame([metrics])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
