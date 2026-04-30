#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace negative flow values with zero in a flowrate CSV."
    )
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: INPUT_stem + _clip_negative_flow.csv",
    )
    parser.add_argument(
        "--flow-cols",
        default="flow1,flow2,flow3,flow4,flow5,flow6",
        help="Comma-separated flow columns to check",
    )
    parser.add_argument(
        "--drop-rows",
        action="store_true",
        help="Drop rows with any negative selected flow instead of clipping negative values to zero.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_clip_negative_flow{input_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    df = pd.read_csv(input_path)
    flow_cols = [col.strip() for col in args.flow_cols.split(",") if col.strip()]
    missing = [col for col in flow_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing flow columns: {missing}. Got: {list(df.columns)}")

    negative = df[flow_cols].lt(0.0)
    if args.drop_rows:
        drop_mask = negative.any(axis=1)
        cleaned = df.loc[~drop_mask].reset_index(drop=True)
        changed_count = int(drop_mask.sum())
        action = "removed rows"
    else:
        cleaned = df.copy()
        cleaned.loc[:, flow_cols] = cleaned.loc[:, flow_cols].clip(lower=0.0)
        changed_count = int(negative.to_numpy().sum())
        action = "clipped values"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"[INFO] input:  {input_path}")
    print(f"[INFO] output: {output_path}")
    print(f"[INFO] rows: {len(df)} -> {len(cleaned)}")
    print(f"[INFO] {action}: {changed_count}")
    for col in flow_cols:
        print(f"[INFO] {col}: negative rows = {int(negative[col].sum())}")


if __name__ == "__main__":
    main()
