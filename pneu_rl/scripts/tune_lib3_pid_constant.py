"""Small gain sweep for the constant-target lib3 PID baseline.

The sweep intentionally uses the same fixed-branch protocol as
quick_viz_lib3_pid_constant.py.  It is meant to find a stable, reproducible
PID reference before comparing RL against it; it is not an RL allocator.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import quick_viz_lib3_pid_constant as quick


def metrics(data: pd.DataFrame, task: str) -> dict[str, float]:
    tail = data.iloc[100:].copy()
    result: dict[str, float] = {"task": task}
    for channel, target in (("pos", "ref_pos"), ("neg", "ref_neg")):
        err = tail[f"act_{channel}"] - tail[target]
        rate = np.gradient(tail[f"act_{channel}"].to_numpy(), quick.DT)
        result[f"{channel}_rmse_tail"] = float(np.sqrt(np.mean(err**2)))
        result[f"{channel}_mae_tail"] = float(np.mean(np.abs(err)))
        result[f"{channel}_tol_tail"] = float(np.mean(np.abs(err) <= 5.0))
        result[f"{channel}_rate_max_tail"] = float(np.max(np.abs(rate)))
        result[f"{channel}_std_tail"] = float(np.std(tail[f"act_{channel}"]))
    result["score"] = (
        result["pos_rmse_tail"] + result["neg_rmse_tail"]
        + 0.10 * result["pos_rate_max_tail"]
        + 0.10 * result["neg_rate_max_tail"]
        + 2.0 * result["pos_std_tail"]
        + 2.0 * result["neg_std_tail"]
    )
    return result


def configure(base: dict, values: dict[str, float]) -> dict:
    cfg = copy.deepcopy(base)
    for channel in ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
        for term in ("kp", "ki"):
            key = f"{channel}_{term}"
            if key in values:
                cfg[channel][term] = values[key]
    cfg["deadband"] = values.get("deadband", cfg["deadband"])
    cfg["integral_limit"] = values.get("integral_limit", cfg["integral_limit"])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_pid_constant_tuning"))
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the chamber allocator fixed and only tune actuator-side gains.
    # The candidates span the region between the old baseline and the
    # oscillatory hand-tuned gains.
    candidates = []
    for pos_kp in (0.012, 0.018, 0.025, 0.035, 0.045):
        for pos_ki in (0.0010, 0.0020, 0.0030):
            candidates.append({
                "act_pos_in_kp": pos_kp,
                "act_pos_in_ki": pos_ki,
                "act_pos_out_kp": 0.035,
                "act_pos_out_ki": 0.0005,
            })
    for neg_kp in (0.012, 0.018, 0.025, 0.035, 0.045):
        for neg_ki in (0.0006, 0.0010, 0.0015, 0.0020):
            candidates.append({
                "act_neg_in_kp": neg_kp,
                "act_neg_in_ki": neg_ki,
                "act_neg_out_kp": 0.022,
                "act_neg_out_ki": 0.0006,
            })

    rows = []
    for index, values in enumerate(candidates):
        cfg = configure(quick.DEFAULT_PID, values)
        quick.DEFAULT_PID = cfg
        for task in ("pos", "neg", "both"):
            ns = argparse.Namespace(
                task=task, target_pos=120.0, target_neg=70.0, steps=args.steps,
                use_baseline_gains=False,
            )
            data = quick.run(ns)
            row = {"candidate": index, **values, **metrics(data, task)}
            rows.append(row)
            print(
                f"candidate={index:02d} task={task:4s} "
                f"tail_rmse=({row['pos_rmse_tail']:.2f},{row['neg_rmse_tail']:.2f}) "
                f"rate=({row['pos_rate_max_tail']:.1f},{row['neg_rate_max_tail']:.1f}) "
                f"score={row['score']:.2f}"
            )

    result = pd.DataFrame(rows)
    result.sort_values(["task", "score"]).to_csv(args.output_dir / "gain_sweep.csv", index=False)
    for task in ("pos", "neg", "both"):
        best = result[result.task == task].sort_values("score").head(5)
        best.to_csv(args.output_dir / f"best_{task}.csv", index=False)
        print(f"\nbest {task}:")
        print(best.to_string(index=False))
    print(f"saved: {args.output_dir / 'gain_sweep.csv'}")


if __name__ == "__main__":
    main()
