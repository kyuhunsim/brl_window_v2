"""Focused per-channel PID tuning for the current lib3 reference.

The first pass uses the stable global setting Kp x2, Ki x0.5.  This sweep
changes only the positive actuator inlet PID, which is the channel responsible
for rising positive actuator pressure.
"""

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

from baseline_lib3 import BASELINE_CONFIG
from evaluate_lib3_matched_pid_and_valve_sweep import plot_pid
from tune_lib3_pid_rate_limited import run_candidate, scaled_pid_config, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_pid_per_channel_tuning"))
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = scaled_pid_config(2.0, 0.5)
    kp_scales = (1.0, 1.5, 2.0, 3.0)
    ki_scales = (0.5, 1.0, 1.5)
    results: list[dict[str, float]] = []
    best_result: dict[str, float] | None = None
    best_data = None

    base_kp = base["act_pos_in"]["kp"]
    base_ki = base["act_pos_in"]["ki"]
    for kp_scale in kp_scales:
        for ki_scale in ki_scales:
            config = copy.deepcopy(base)
            config["act_pos_in"]["kp"] = base_kp * kp_scale
            config["act_pos_in"]["ki"] = base_ki * ki_scale
            data = run_candidate(
                1.0,
                1.0,
                blocks=args.blocks,
                block_steps=args.block_steps,
                carry=True,
                pid_config=config,
            )
            metrics = score(data)
            result = {
                "act_pos_in_kp_relative_to_global_best": kp_scale,
                "act_pos_in_ki_relative_to_global_best": ki_scale,
                **metrics,
            }
            results.append(result)
            print(
                f"act_pos_in kp={kp_scale:g} ki={ki_scale:g} "
                f"POS={metrics['pos_rmse_kpa']:.3f} "
                f"NEG={metrics['neg_rmse_kpa']:.3f} "
                f"POS_rate={metrics['act_pos_rate_abs_max_kpa_s']:.2f} "
                f"score={metrics['score']:.3f}"
            )
            if best_result is None or result["score"] < best_result["score"]:
                best_result = result
                best_data = data

    ranked = pd.DataFrame(results).sort_values("score")
    ranked.to_csv(args.output_dir / "act_pos_in_sweep.csv", index=False)
    if best_data is not None:
        best_data.to_csv(args.output_dir / "best_trajectory.csv", index=False)
        plot_pid(best_data, args.output_dir / "best_trajectory.png", episode=0)

    summary = {
        "base_global_setting": {
            "all_kp_scale": 2.0,
            "all_ki_scale": 0.5,
            "all_kd_scale": 1.0,
        },
        "tuned_channel": "act_pos_in",
        "protocol": {
            "blocks": args.blocks,
            "block_steps": args.block_steps,
            "carry": True,
            "total_steps": args.blocks * args.block_steps,
            "freq_hz": 50.0,
            "delay_sec": 0.1,
            "scale_low": 0.8,
        },
        "original_pid": BASELINE_CONFIG["pid"]["act_pos_in"],
        "best": best_result,
        "ranked_results": results,
    }
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print("\nbest:")
    print(yaml.safe_dump(best_result, sort_keys=False))
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
