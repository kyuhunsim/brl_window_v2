#!/usr/bin/env python3
"""Aggregate per-member reset-bank evaluations into one row per model.

The input is the ``summary.yaml`` emitted by ``summarize_screening_batch.py``
after evaluating a single model on every member of an optimized reference
bank.  It reports central and tail tracking error, rather than allowing one
easy member to represent the whole training distribution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def finite(values: list[Any]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))])


def aggregate(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = list(yaml.safe_load(handle) or [])
    if not rows:
        raise ValueError(f"empty bank summary: {path}")
    model = str(rows[0].get("model", path.parent.name))
    result: dict[str, Any] = {
        "model": model,
        "variant": rows[0].get("variant"),
        "members": len(rows),
        "all_reference_matches": bool(all(row.get("reference_content_match", True) for row in rows)),
        "source_summary": str(path),
    }
    for axis in ("pos", "neg"):
        rmse = finite([row.get(f"{axis}_rmse") for row in rows])
        success = finite([row.get(f"{axis}_success_ratio") for row in rows])
        if len(rmse):
            result[f"{axis}_rmse_mean"] = float(np.mean(rmse))
            result[f"{axis}_rmse_median"] = float(np.median(rmse))
            result[f"{axis}_rmse_p90"] = float(np.quantile(rmse, 0.90))
            result[f"{axis}_rmse_worst"] = float(np.max(rmse))
        if len(success):
            result[f"{axis}_success_ratio_mean"] = float(np.mean(success))
    reward = finite([row.get("reward_per_step") for row in rows])
    if len(reward):
        result["reward_per_step_mean"] = float(np.mean(reward))
        result["reward_per_step_p10"] = float(np.quantile(reward, 0.10))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [aggregate(path) for path in args.bank_summary]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "variant", "members", "all_reference_matches",
        "pos_rmse_mean", "pos_rmse_median", "pos_rmse_p90", "pos_rmse_worst",
        "neg_rmse_mean", "neg_rmse_median", "neg_rmse_p90", "neg_rmse_worst",
        "pos_success_ratio_mean", "neg_success_ratio_mean",
        "reward_per_step_mean", "reward_per_step_p10", "source_summary",
    ]
    csv_rows = [{field: row.get(field) for field in fields} for row in rows]
    with (args.output_dir / "reference_bank_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    with (args.output_dir / "reference_bank_aggregate.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(csv_rows, handle, sort_keys=False)

    labels = [str(row["variant"] or row["model"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, pressure in zip(axes, ("pos", "neg")):
        median = [row.get(f"{pressure}_rmse_median", np.nan) for row in rows]
        p90 = [row.get(f"{pressure}_rmse_p90", np.nan) for row in rows]
        axis.bar(x - width / 2, median, width, label="median")
        axis.bar(x + width / 2, p90, width, label="p90")
        axis.set_title(f"{pressure.upper()} RMSE across reference bank")
        axis.set_ylabel("kPa")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "reference_bank_aggregate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] wrote {args.output_dir / 'reference_bank_aggregate.yaml'}")


if __name__ == "__main__":
    main()
