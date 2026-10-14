"""Tune the existing lib3 reserve/directional PID on the current reference.

This is intentionally a small gain sweep, not a new controller.  The plant
and reference are continuous across 10 x 512-step blocks by default so that
the check matches the intended carry protocol.
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

from baseline_lib3 import BASELINE_CONFIG, chamber_allocator, pid_allocator
from evaluate_lib3_matched_pid_and_valve_sweep import (
    ATM,
    REF_KWARGS,
    add_rates,
    make_reference,
    make_sim,
    plot_pid,
    pid_metrics,
)
from pneu_ref.pneu_ref import PneuRef


def scaled_pid_config(kp_scale: float, ki_scale: float) -> dict:
    config = copy.deepcopy(BASELINE_CONFIG["pid"])
    for value in config.values():
        if isinstance(value, dict):
            value["kp"] *= kp_scale
            value["ki"] *= ki_scale
    return config


def run_candidate(
    kp_scale: float,
    ki_scale: float,
    *,
    blocks: int,
    block_steps: int,
    carry: bool,
    pid_config: dict | None = None,
    active_min_raw: dict[int, float] | None = None,
) -> pd.DataFrame:
    sim = make_sim()
    reference = PneuRef(
        make_reference(),
        num_prev=10,
        num_pred=15,
        ctrl_freq=50.0,
    )
    pid_config = (
        scaled_pid_config(kp_scale, ki_scale)
        if pid_config is None
        else copy.deepcopy(pid_config)
    )
    previous_error = np.zeros(4, dtype=np.float64)
    integral_error = np.zeros(4, dtype=np.float64)
    rows: list[dict[str, float]] = []
    dt = 1.0 / 50.0
    total_step = 0

    for block in range(blocks):
        if block == 0 or not carry:
            sim.set_init_press(ATM, ATM, ATM, ATM)
            reference.reset(initial_state=np.array([ATM, ATM], dtype=np.float64))
            previous_error = np.zeros(4, dtype=np.float64)
            integral_error = np.zeros(4, dtype=np.float64)

        for block_step in range(block_steps):
            local_step = block_step if not carry else total_step
            curr_time = local_step * dt
            goal_traj = reference.get_ref(curr_time).reshape(-1, 2)
            goal = np.asarray(goal_traj[10], dtype=np.float64)

            chamber_ctrl, chamber_target = chamber_allocator(
                sim.obs[0:2],
                goal,
                mode="reserve",
                chamber_raw=BASELINE_CONFIG["chamber"]["raw"],
                margin=35.0,
                kp=0.04,
                deadband=1.0,
                min_open=0.15,
                max_open=0.85,
                ch_pos_min=ATM,
                ch_pos_max=200.0,
                ch_neg_min=15.0,
                ch_neg_max=ATM,
            )
            action, previous_error, integral_error = pid_allocator(
                sim.obs[2:4],
                goal,
                previous_error,
                integral_error,
                dt=dt,
                pid_cfg=pid_config,
                chamber_ctrl=chamber_ctrl,
            )
            if active_min_raw:
                for index, minimum in active_min_raw.items():
                    if previous_error[index] > float(pid_config["deadband"]):
                        action[index] = max(action[index], float(minimum))
            obs, _ = sim.observe(action, goal=goal)
            valve_debug = sim.get_valve_debug()
            row = {
                "block": float(block + 1),
                "block_step": float(block_step),
                "step": float(total_step),
                "time": float((total_step + 1) * dt),
                "ref_pos": float(goal[0]),
                "ref_neg": float(goal[1]),
                "chamber_pos": float(obs[1]),
                "chamber_neg": float(obs[2]),
                "act_pos": float(obs[3]),
                "act_neg": float(obs[4]),
                "chamber_target_pos": float(chamber_target[0]),
                "chamber_target_neg": float(chamber_target[1]),
            }
            for index, value in enumerate(action):
                row[f"ctrl{index + 1}_physical"] = float(0.8 + 0.2 * value)
            # Preserve the hidden actuator quantities for diagnosis.  These
            # columns are deliberately not fed to the PID/RL policy.
            for key, value in valve_debug.items():
                row[key] = float(value)
            rows.append(row)
            total_step += 1

    return pd.DataFrame(rows)


def score(data: pd.DataFrame) -> dict[str, float]:
    metrics = pid_metrics(data)
    rates = add_rates(data)
    pos_excess = np.maximum(np.abs(rates["rate_act_pos"]) - 32.0, 0.0)
    neg_excess = np.maximum(np.abs(rates["rate_act_neg"]) - 9.5, 0.0)
    metrics["rate_excess_pos_kpa_s"] = float(np.max(pos_excess))
    metrics["rate_excess_neg_kpa_s"] = float(np.max(neg_excess))
    metrics["score_rmse_sum"] = metrics["pos_rmse_kpa"] + metrics["neg_rmse_kpa"]
    metrics["score"] = (
        metrics["score_rmse_sum"]
        + 2.0 * metrics["rate_excess_pos_kpa_s"]
        + 2.0 * metrics["rate_excess_neg_kpa_s"]
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_pid_gain_tuning"))
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=512)
    parser.add_argument("--no-carry", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kp_scales = (0.5, 1.0, 2.0, 4.0)
    ki_scales = (0.5, 1.0, 2.0)
    results: list[dict[str, float]] = []
    best_data: pd.DataFrame | None = None
    best_result: dict[str, float] | None = None

    for kp_scale in kp_scales:
        for ki_scale in ki_scales:
            data = run_candidate(
                kp_scale,
                ki_scale,
                blocks=args.blocks,
                block_steps=args.block_steps,
                carry=not args.no_carry,
            )
            result = {
                "kp_scale": kp_scale,
                "ki_scale": ki_scale,
                "carry": float(not args.no_carry),
                **score(data),
            }
            results.append(result)
            print(
                f"kp={kp_scale:g} ki={ki_scale:g} "
                f"POS={result['pos_rmse_kpa']:.3f} "
                f"NEG={result['neg_rmse_kpa']:.3f} "
                f"tol=({result['pos_tolerance_fraction']:.3f},"
                f"{result['neg_tolerance_fraction']:.3f}) "
                f"score={result['score']:.3f}"
            )
            if best_result is None or result["score"] < best_result["score"]:
                best_result = result
                best_data = data

    result_df = pd.DataFrame(results).sort_values("score")
    result_df.to_csv(args.output_dir / "gain_sweep.csv", index=False)
    if best_data is not None:
        best_data.to_csv(args.output_dir / "best_trajectory.csv", index=False)
        plot_pid(best_data, args.output_dir / "best_trajectory.png", episode=0)
    summary = {
        "reference": {"kind": "rate_limited_random", "kwargs": REF_KWARGS},
        "protocol": {
            "blocks": args.blocks,
            "block_steps": args.block_steps,
            "carry": not args.no_carry,
            "freq_hz": 50.0,
            "delay_sec": 0.1,
            "scale_low": 0.8,
        },
        "baseline_pid": BASELINE_CONFIG["pid"],
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
