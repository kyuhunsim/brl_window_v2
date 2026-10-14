#!/usr/bin/env python3
"""Compute offline TD(lambda) discrepancy from lib4 validation traces.

Examples
--------
python pneu_rl/scripts/analyze_lambda_discrepancy.py \
  pneu_rl/exp/.../lib4_history_rate_reserve_seed0/member_00/validation.csv

Directories are searched recursively for ``validation.csv``.  The evaluator
now writes a deterministic ``value`` column.  Comparing files produced by
different policies is policy-dependent; use this tool as a within-policy or
common-frozen-policy representation diagnostic, not as a standalone ranking
of controllers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for source_dir in (
    ROOT / "pneu_rl" / "src",
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_utils" / "src",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_rl.lambda_discrepancy import discrepancy_from_columns


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            paths.extend(sorted(path.rglob("validation.csv")))
        elif path.is_file():
            paths.append(path)
        else:
            matches = sorted(path.parent.glob(path.name))
            paths.extend(match for match in matches if match.is_file())
    # Preserve input order while removing duplicates.
    return list(dict.fromkeys(paths))


def read_trace(path: Path, reward_column: str, value_column: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    missing = {reward_column, value_column} - set(rows[0])
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)}; re-run evaluate_lib4_screening.py "
            "after the value-column update"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--reward-column", default="reward")
    parser.add_argument("--value-column", default="value")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-low", type=float, default=0.0)
    parser.add_argument("--lambda-high", type=float, default=0.95)
    parser.add_argument("--terminal-value", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = resolve_inputs(args.inputs)
    if not paths:
        parser.error("no validation.csv traces found")
    reports: list[dict[str, object]] = []
    for path in paths:
        rows = read_trace(path, args.reward_column, args.value_column)
        report = discrepancy_from_columns(
            rows,
            reward_column=args.reward_column,
            value_column=args.value_column,
            gamma=args.gamma,
            lambda_low=args.lambda_low,
            lambda_high=args.lambda_high,
            terminal_value=args.terminal_value,
        )
        report = {"trace": str(path), **report}
        reports.append(report)

    # A directory input normally contains model/member_##/validation.csv.
    # Report a compact mean/std per model as well as the individual members.
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for report in reports:
        trace = Path(str(report["trace"]))
        model_key = trace.parents[1].name if trace.parent.name.startswith("member_") else trace.parent.name
        grouped[model_key].append(report)
    aggregate: dict[str, dict[str, float | int]] = {}
    metric_names = (
        "mean_abs",
        "rmse",
        "p95_abs",
        "max_abs",
        "normalized_mean_abs",
    )
    for model_key, model_reports in sorted(grouped.items()):
        summary: dict[str, float | int] = {"n_traces": len(model_reports)}
        for metric in metric_names:
            values = [float(item[metric]) for item in model_reports]
            summary[f"{metric}_mean"] = float(sum(values) / len(values))
            summary[f"{metric}_std"] = float(
                (sum((value - summary[f"{metric}_mean"]) ** 2 for value in values) / len(values)) ** 0.5
            )
        aggregate[model_key] = summary

    config = vars(args).copy()
    config["inputs"] = [str(p) for p in args.inputs]
    if config.get("output") is not None:
        config["output"] = str(config["output"])
    payload = {"config": config, "reports": reports, "aggregate_by_model": aggregate}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
