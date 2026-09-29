from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import make_base_features
from .io import load_dataset, write_submission_xlsx
from .scoring import framework_text, score_variant
from .validate import validate_predictions


def build_submission(
    data_path: str | Path,
    template_path: str | Path,
    output_xlsx: str | Path,
    output_csv: str | Path | None = None,
    variant: str = "attempt8_plus_travelpremium",
) -> pd.DataFrame:
    raw = load_dataset(data_path)
    features = make_base_features(raw)
    scored = score_variant(features, variant=variant)
    predictions = scored[["ID", "Prediction"]].copy()
    validate_predictions(raw, predictions)

    write_submission_xlsx(
        predictions=predictions,
        template_path=template_path,
        output_path=output_xlsx,
        framework=framework_text(variant),
    )
    if output_csv:
        predictions.to_csv(output_csv, index=False)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AMEX R1 profitability submission workbook.")
    parser.add_argument("--data", required=True, help="Path to R1dataset.csv")
    parser.add_argument("--template", required=True, help="Path to submission_template.xlsx")
    parser.add_argument("--output-xlsx", required=True, help="Output XLSX path")
    parser.add_argument("--output-csv", default=None, help="Optional predictions CSV path")
    parser.add_argument(
        "--variant",
        default="attempt8_plus_travelpremium",
        choices=["baseline_086", "attempt7_refined_eb", "attempt8_rebalanced", "attempt8_plus_travelpremium"],
    )
    args = parser.parse_args()
    Path(args.output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    build_submission(args.data, args.template, args.output_xlsx, args.output_csv, args.variant)
    print(f"Wrote {args.output_xlsx}")
    if args.output_csv:
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
