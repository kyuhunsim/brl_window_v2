"""Test a valve-aware minimum opening for the positive inlet PID channel."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_lib3_matched_pid_and_valve_sweep import add_rates, plot_pid
from tune_lib3_pid_rate_limited import run_candidate


PID = {
    "act_pos_in": {"kp": 0.030, "ki": 0.00100, "kd": 0.001},
    "act_pos_out": {"kp": 0.050, "ki": 0.00025, "kd": 0.001},
    "act_neg_in": {"kp": 0.038, "ki": 0.00135, "kd": 0.001},
    "act_neg_out": {"kp": 0.075, "ki": 0.00060, "kd": 0.001},
    "deadband": 0.5,
    "integral_limit": 100.0,
    "max_open": 1.0,
}


def metrics(data: pd.DataFrame) -> dict[str, float]:
    d = add_rates(data)
    rising = d.rate_ref_pos > 1.0
    pos_err = d.act_pos - d.ref_pos
    rise_rate_err = d.loc[rising, "rate_act_pos"] - d.loc[rising, "rate_ref_pos"]
    return {
        "pos_rmse_kpa": float(np.sqrt(np.mean(pos_err**2))),
        "neg_rmse_kpa": float(np.sqrt(np.mean((d.act_neg - d.ref_neg) ** 2))),
        "rise_pos_mae_kpa": float(np.mean(np.abs(pos_err[rising]))),
        "rise_rate_mae_kpa_s": float(np.mean(np.abs(rise_rate_err))),
        "rise_actual_rate_mean_kpa_s": float(d.loc[rising, "rate_act_pos"].mean()),
        "rise_reference_rate_mean_kpa_s": float(d.loc[rising, "rate_ref_pos"].mean()),
        "rise_u3_mean_physical": float(d.loc[rising, "ctrl3_physical"].mean()),
        "rise_u3_ge_092_fraction": float(np.mean(d.loc[rising, "ctrl3_physical"] >= 0.92)),
        "act_pos_rate_abs_max_kpa_s": float(np.max(np.abs(d.rate_act_pos))),
        "act_neg_rate_abs_max_kpa_s": float(np.max(np.abs(d.rate_act_neg))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_pid_valve_aware_opening"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    physical_minima = (0.90, 0.92, 0.94)
    results = []
    best_data = None
    best = None
    for physical_min in physical_minima:
        raw_min = (physical_min - 0.8) / 0.2
        data = run_candidate(
            1.0,
            1.0,
            blocks=10,
            block_steps=512,
            carry=True,
            pid_config=copy.deepcopy(PID),
            active_min_raw={2: raw_min},
        )
        result = {"act_pos_in_min_physical": physical_min, **metrics(data)}
        results.append(result)
        print(yaml.safe_dump(result, sort_keys=False))
        # Prefer rising-segment tracking, with a strong penalty over the
        # screening POS rate limit of 32 kPa/s.
        score = result["rise_pos_mae_kpa"] + 0.25 * result["rise_rate_mae_kpa_s"]
        score += 2.0 * max(result["act_pos_rate_abs_max_kpa_s"] - 32.0, 0.0)
        result["score"] = score
        if best is None or score < best["score"]:
            best = result
            best_data = data

    result_df = pd.DataFrame(results).sort_values("score")
    result_df.to_csv(args.output_dir / "minimum_opening_sweep.csv", index=False)
    if best_data is not None:
        best_data.to_csv(args.output_dir / "best_trajectory.csv", index=False)
        plot_pid(best_data, args.output_dir / "best_trajectory.png", episode=0)
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"best": best, "results": results}, handle, sort_keys=False)
    print("best:")
    print(yaml.safe_dump(best, sort_keys=False))
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
