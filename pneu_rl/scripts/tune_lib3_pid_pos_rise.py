"""Tune POS inlet PID using positive-reference rising segments as the objective."""

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


FINAL_PID = {
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
    rising = d["rate_ref_pos"] > 1.0
    if not np.any(rising):
        raise RuntimeError("reference has no positive rising samples")
    pos_error = d["act_pos"] - d["ref_pos"]
    rate_error = d["rate_act_pos"] - d["rate_ref_pos"]
    rise_pos_error = pos_error[rising]
    rise_rate_error = rate_error[rising]
    actual_rise_rate = d.loc[rising, "rate_act_pos"].to_numpy()
    desired_rise_rate = d.loc[rising, "rate_ref_pos"].to_numpy()
    # Primary objective is the rising segment.  The rate penalty prevents a
    # high-integral solution from winning merely by oscillating through the
    # target.
    rise_mae = float(np.mean(np.abs(rise_pos_error)))
    rise_rate_mae = float(np.mean(np.abs(rise_rate_error)))
    rate_excess = float(max(np.max(np.abs(d["rate_act_pos"])) - 32.0, 0.0))
    score = 2.0 * rise_mae + 0.35 * rise_rate_mae + 2.0 * rate_excess
    return {
        "rising_samples": int(np.sum(rising)),
        "rise_pos_mae_kpa": rise_mae,
        "rise_rate_mae_kpa_s": rise_rate_mae,
        "rise_actual_rate_mean_kpa_s": float(np.mean(actual_rise_rate)),
        "rise_reference_rate_mean_kpa_s": float(np.mean(desired_rise_rate)),
        "rise_actual_positive_fraction": float(np.mean(actual_rise_rate > 0.0)),
        "pos_rmse_kpa": float(np.sqrt(np.mean(pos_error**2))),
        "neg_rmse_kpa": float(np.sqrt(np.mean((d["act_neg"] - d["ref_neg"]) ** 2))),
        "act_pos_rate_abs_max_kpa_s": float(np.max(np.abs(d["rate_act_pos"]))),
        "act_neg_rate_abs_max_kpa_s": float(np.max(np.abs(d["rate_act_neg"]))),
        "score_rising": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_pid_pos_rise_tuning"))
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kp_values = (0.030, 0.033, 0.036, 0.039, 0.042)
    ki_values = (0.00075, 0.00100, 0.00125)
    results: list[dict[str, float]] = []
    best = None
    best_data = None

    for kp in kp_values:
        for ki in ki_values:
            config = copy.deepcopy(FINAL_PID)
            config["act_pos_in"] = {"kp": kp, "ki": ki, "kd": 0.001}
            data = run_candidate(
                1.0,
                1.0,
                blocks=args.blocks,
                block_steps=args.block_steps,
                carry=True,
                pid_config=config,
            )
            result = {"act_pos_in_kp": kp, "act_pos_in_ki": ki, **metrics(data)}
            results.append(result)
            print(
                f"kp={kp:.5f} ki={ki:.5f} "
                f"rise_mae={result['rise_pos_mae_kpa']:.3f} "
                f"rise_rate_mae={result['rise_rate_mae_kpa_s']:.3f} "
                f"actual_rise={result['rise_actual_rate_mean_kpa_s']:.3f} "
                f"max_rate={result['act_pos_rate_abs_max_kpa_s']:.2f} "
                f"score={result['score_rising']:.3f}"
            )
            if best is None or result["score_rising"] < best["score_rising"]:
                best = result
                best_data = data

    ranked = pd.DataFrame(results).sort_values("score_rising")
    ranked.to_csv(args.output_dir / "pos_rise_sweep.csv", index=False)
    if best_data is not None:
        best_data.to_csv(args.output_dir / "best_trajectory.csv", index=False)
        plot_pid(best_data, args.output_dir / "best_trajectory.png", episode=0)
    summary = {
        "objective": "positive-reference-rising-segments",
        "protocol": {
            "blocks": args.blocks,
            "block_steps": args.block_steps,
            "carry": True,
            "total_steps": args.blocks * args.block_steps,
            "rise_mask": "rate_ref_pos > 1.0 kPa/s",
        },
        "base_pid": FINAL_PID,
        "best": best,
    }
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print("\nbest:")
    print(yaml.safe_dump(best, sort_keys=False))
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
