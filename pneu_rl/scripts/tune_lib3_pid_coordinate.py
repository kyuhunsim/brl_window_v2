"""Coordinate-descent tuning of the four directional lib3 PID channels."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

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
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_pid_coordinate_tuning"))
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Start from the rate-safe global setting and the already measured
    # positive-inlet improvement: act_pos_in Kp x1.5, Ki x0.5 relative to it.
    config = scaled_pid_config(2.0, 0.5)
    config["act_pos_in"]["kp"] *= 1.5
    config["act_pos_in"]["ki"] *= 0.5

    channels = ("act_pos_out", "act_neg_in", "act_neg_out")
    kp_scales = (0.5, 1.0, 1.5, 2.0)
    ki_scales = (0.5, 1.0, 1.5)
    all_results: dict[str, list[dict[str, float]]] = {}
    selected: dict[str, dict[str, float]] = {}

    for channel in channels:
        base_channel = copy.deepcopy(config[channel])
        channel_results: list[dict[str, float]] = []
        best_result = None
        best_config = None
        for kp_scale in kp_scales:
            for ki_scale in ki_scales:
                candidate = copy.deepcopy(config)
                candidate[channel]["kp"] = base_channel["kp"] * kp_scale
                candidate[channel]["ki"] = base_channel["ki"] * ki_scale
                data = run_candidate(
                    1.0,
                    1.0,
                    blocks=args.blocks,
                    block_steps=args.block_steps,
                    carry=True,
                    pid_config=candidate,
                )
                metrics = score(data)
                result = {
                    "channel": channel,
                    "kp_relative": kp_scale,
                    "ki_relative": ki_scale,
                    **metrics,
                }
                channel_results.append(result)
                print(
                    f"{channel} kp={kp_scale:g} ki={ki_scale:g} "
                    f"POS={metrics['pos_rmse_kpa']:.3f} "
                    f"NEG={metrics['neg_rmse_kpa']:.3f} "
                    f"rate=({metrics['act_pos_rate_abs_max_kpa_s']:.2f},"
                    f"{metrics['act_neg_rate_abs_max_kpa_s']:.2f}) "
                    f"score={metrics['score']:.3f}"
                )
                if best_result is None or result["score"] < best_result["score"]:
                    best_result = result
                    best_config = candidate
        assert best_result is not None and best_config is not None
        config = best_config
        selected[channel] = best_result
        all_results[channel] = channel_results
        print(f"selected {channel}: kp={best_result['kp_relative']:g} ki={best_result['ki_relative']:g}")

    final_data = run_candidate(
        1.0,
        1.0,
        blocks=args.blocks,
        block_steps=args.block_steps,
        carry=True,
        pid_config=config,
    )
    final_data.to_csv(args.output_dir / "final_trajectory.csv", index=False)
    plot_pid(final_data, args.output_dir / "final_trajectory.png", episode=0)
    for channel, results in all_results.items():
        pd.DataFrame(results).sort_values("score").to_csv(
            args.output_dir / f"{channel}_sweep.csv", index=False
        )

    summary = {
        "protocol": {
            "blocks": args.blocks,
            "block_steps": args.block_steps,
            "carry": True,
            "total_steps": args.blocks * args.block_steps,
            "freq_hz": 50.0,
            "delay_sec": 0.1,
            "scale_low": 0.8,
        },
        "starting_pid": BASELINE_CONFIG["pid"],
        "final_pid": config,
        "selected_channels": selected,
        "all_results": all_results,
        "final_metrics": score(final_data),
    }
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print("\nfinal metrics:")
    print(yaml.safe_dump(summary["final_metrics"], sort_keys=False))
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
