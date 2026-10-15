"""Matched lib3 PID evaluation and valve effective-opening sweep.

The PID run uses the same rate-limited reference configuration as the current
lib3 SAC screening models.  The sweep is an open-loop diagnostic: each valve
is varied over the physical command range while the other valves remain at
the configured lower command.  It reports simulator valve-debug quantities,
not just the commanded signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim3 import PneuSim
from pneu_ref.feasible_random_ref import FeasibleRandomRef
from pneu_ref.pneu_ref import PneuRef

# Reuse the committed PID allocator and gains used by lib3_pid_baseline_40s.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from baseline_lib3 import ATM, BASELINE_CONFIG, chamber_allocator, pid_allocator


REF_KWARGS = dict(
    pos_min=105.0,
    pos_max=150.0,
    neg_min=55.0,
    neg_max=85.0,
    pos_rate_up=12.0,
    pos_rate_down=12.0,
    neg_rate_up=6.0,
    neg_rate_down=6.0,
    hold_min=1.5,
    hold_max=4.0,
    min_transition=2.5,
    max_frequency_hz=0.2,
    initial_pos=ATM,
    initial_neg=ATM,
    initial_hold=0.0,
    min_pos_delta=5.0,
    min_neg_delta=2.0,
    seed=61098,
)

VALVES = (
    "ch_pos",
    "ch_neg",
    "act_pos_in",
    "act_pos_out",
    "act_neg_in",
    "act_neg_out",
)

SWEEP_STATES = (
    (ATM, ATM, ATM, ATM),
    (140.0, 40.0, 130.0, 70.0),
    (120.0, 20.0, 115.0, 60.0),
    (180.0, 80.0, 145.0, 80.0),
)


def make_sim() -> PneuSim:
    return PneuSim(
        freq=50.0,
        delay=0.1,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=0.8,
        scale_high=1.0,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )


def make_reference() -> FeasibleRandomRef:
    return FeasibleRandomRef(**REF_KWARGS)


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dt = 1.0 / 50.0
    for name in ("ref_pos", "ref_neg", "act_pos", "act_neg"):
        result[f"rate_{name}"] = np.gradient(result[name].to_numpy(), dt)
    return result


def pid_metrics(df: pd.DataFrame, tolerance: float = 5.0) -> dict[str, float]:
    data = add_rates(df)
    pos_err = data["act_pos"] - data["ref_pos"]
    neg_err = data["act_neg"] - data["ref_neg"]
    return {
        "steps": int(len(data)),
        "duration_sec": float(len(data) / 50.0),
        "pos_rmse_kpa": float(np.sqrt(np.mean(pos_err**2))),
        "neg_rmse_kpa": float(np.sqrt(np.mean(neg_err**2))),
        "pos_mae_kpa": float(np.mean(np.abs(pos_err))),
        "neg_mae_kpa": float(np.mean(np.abs(neg_err))),
        "pos_tolerance_fraction": float(np.mean(np.abs(pos_err) <= tolerance)),
        "neg_tolerance_fraction": float(np.mean(np.abs(neg_err) <= tolerance)),
        "ref_pos_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_ref_pos"]))),
        "ref_neg_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_ref_neg"]))),
        "act_pos_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_act_pos"]))),
        "act_neg_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_act_neg"]))),
        "chamber_pos_max_kpa": float(data["chamber_pos"].max()),
        "chamber_neg_min_kpa": float(data["chamber_neg"].min()),
    }


def run_pid_episode(
    sim: PneuSim,
    ref: PneuRef,
    steps: int,
) -> pd.DataFrame:
    sim.set_init_press(ATM, ATM, ATM, ATM)
    ref.reset(initial_state=np.array([ATM, ATM], dtype=np.float64))
    prev_err = np.zeros(4, dtype=np.float64)
    integral_err = np.zeros(4, dtype=np.float64)
    rows: list[dict[str, float]] = []
    dt = 1.0 / 50.0

    for step in range(steps):
        curr_time = step * dt
        goal_traj = ref.get_ref(curr_time).reshape(-1, 2)
        # Match the current MPOBS environment: num_prev=10 and the actuator
        # tracking target is the current entry at index 10.
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
        action, prev_err, integral_err = pid_allocator(
            sim.obs[2:4],
            goal,
            prev_err,
            integral_err,
            dt=dt,
            pid_cfg=BASELINE_CONFIG["pid"],
            chamber_ctrl=chamber_ctrl,
        )
        obs, _ = sim.observe(action, goal=goal)
        debug = sim.get_valve_debug()
        row: dict[str, float] = {
            "step": float(step),
            "time": float((step + 1) * dt),
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
            row[f"ctrl{index + 1}_raw"] = float(value)
            row[f"ctrl{index + 1}_physical"] = float(0.8 + 0.2 * value)
        for key, value in debug.items():
            row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pid(df: pd.DataFrame, path: Path, episode: int) -> None:
    data = add_rates(df)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(data.time, data.ref_pos, "k", label="REF POS")
    axes[0].plot(data.time, data.act_pos, "r", label="ACT POS")
    axes[0].plot(data.time, data.ref_neg, "k--", label="REF NEG")
    axes[0].plot(data.time, data.act_neg, "b", label="ACT NEG")
    axes[0].set_ylabel("Pressure [kPa]")
    axes[0].legend(ncol=4)
    axes[1].plot(data.time, data.rate_ref_pos, "r", label="dREF POS")
    axes[1].plot(data.time, data.rate_act_pos, "r--", label="dACT POS")
    axes[1].plot(data.time, data.rate_ref_neg, "b", label="dREF NEG")
    axes[1].plot(data.time, data.rate_act_neg, "b--", label="dACT NEG")
    axes[1].set_ylabel("Rate [kPa/s]")
    axes[1].legend(ncol=4)
    axes[2].plot(data.time, data.chamber_pos, label="Ch POS")
    axes[2].plot(data.time, data.chamber_neg, label="Ch NEG")
    axes[2].plot(data.time, data.chamber_target_pos, "--", label="Ch POS target")
    axes[2].plot(data.time, data.chamber_target_neg, "--", label="Ch NEG target")
    axes[2].set_ylabel("Chamber [kPa]")
    axes[2].legend(ncol=4)
    for index in range(6):
        axes[3].plot(data.time, data[f"ctrl{index + 1}_physical"], label=f"u{index + 1}")
    axes[3].set_ylabel("Physical command")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylim(0.78, 1.02)
    axes[3].legend(ncol=6)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"lib3 matched rate-limited PID | episode {episode}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_pid(output_dir: Path, episodes: int, steps: int) -> None:
    sim = make_sim()
    # Match the current MPOBS+CAPS environment's PneuRef buffering.  The
    # wrapper is intentionally constructed once and reset between episodes so
    # its RNG stream advances exactly as the evaluation environment does.
    ref = PneuRef(make_reference(), num_prev=10, num_pred=15, ctrl_freq=50.0)
    metrics_rows = []
    for episode in range(1, episodes + 1):
        data = run_pid_episode(sim, ref, steps)
        data.to_csv(output_dir / f"pid_episode_{episode:02d}.csv", index=False)
        plot_pid(data, output_dir / f"pid_episode_{episode:02d}.png", episode)
        metrics_rows.append({"episode": episode, **pid_metrics(data)})
    summary = {
        "controller": "reserve_directional_pid",
        "reference": {"kind": "rate_limited_random", "kwargs": REF_KWARGS},
        "conditions": {
            "lib": 3,
            "episodes": episodes,
            "steps": steps,
            "freq_hz": 50.0,
            "delay_sec": 0.1,
            "scale_low": 0.8,
            "scale_high": 1.0,
            "initial_pressure_kpa": ATM,
            "pid_enabled_in_sim": False,
        },
        "episodes": metrics_rows,
        "mean": {
            key: float(np.mean([row[key] for row in metrics_rows]))
            for key in metrics_rows[0]
            if key != "episode"
        },
    }
    with (output_dir / "pid_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(yaml.safe_dump(summary, sort_keys=False))


def run_sweep(output_dir: Path, points: int, warmup_sec: float, measure_sec: float) -> None:
    commands = np.linspace(0.8, 1.0, points)
    rows: list[dict[str, float | str]] = []
    warmup_steps = max(int(round(warmup_sec * 50.0)), 1)
    measure_steps = max(int(round(measure_sec * 50.0)), 2)
    sim = make_sim()
    for state_id, initial_state in enumerate(SWEEP_STATES):
        for valve_index, valve_name in enumerate(VALVES):
            for physical in commands:
                action = np.zeros(6, dtype=np.float64)
                action[valve_index] = (physical - 0.8) / 0.2
                sim.set_init_press(*initial_state)
                for _ in range(warmup_steps):
                    sim.observe(action)
                debug_values = []
                pressure_values = []
                for _ in range(measure_steps):
                    obs, _ = sim.observe(action)
                    debug_values.append(sim.get_valve_debug())
                    pressure_values.append(obs.copy())
                debug_df = pd.DataFrame(debug_values)
                pressure = np.asarray(pressure_values)
                rate = (pressure[-1, 3:5] - pressure[0, 3:5]) / measure_sec
                row: dict[str, float | str] = {
                    "state_id": float(state_id),
                    "valve_index": float(valve_index),
                    "valve": valve_name,
                    "initial_ch_pos": float(initial_state[0]),
                    "initial_ch_neg": float(initial_state[1]),
                    "initial_act_pos": float(initial_state[2]),
                    "initial_act_neg": float(initial_state[3]),
                    "physical_command": float(physical),
                    "normalized_command": float(action[valve_index]),
                    "pos_rate_kpa_s": float(rate[0]),
                    "neg_rate_kpa_s": float(rate[1]),
                }
                for name in ("u_eff", "force_net", "area_eff", "q_pred_lpm", "mdot"):
                    row[f"mean_{name}"] = float(debug_df[f"{valve_name}_{name}"].mean())
                    row[f"max_{name}"] = float(debug_df[f"{valve_name}_{name}"].max())
                rows.append(row)
    data = pd.DataFrame(rows)
    data.to_csv(output_dir / "valve_sweep.csv", index=False)

    threshold_rows = []
    for (state_id, valve), group in data.groupby(["state_id", "valve"]):
        group = group.sort_values("physical_command")
        max_q = float(group["mean_q_pred_lpm"].max())
        max_mdot = float(group["mean_mdot"].max())
        threshold_rows.append(
            {
                "state_id": int(state_id),
                "valve": valve,
                "max_mean_q_pred_lpm": max_q,
                "command_at_10pct_qmax": float(
                    group.loc[group.mean_q_pred_lpm >= 0.10 * max_q, "physical_command"].iloc[0]
                ) if max_q > 0.0 and np.any(group.mean_q_pred_lpm >= 0.10 * max_q) else np.nan,
                "command_at_50pct_qmax": float(
                    group.loc[group.mean_q_pred_lpm >= 0.50 * max_q, "physical_command"].iloc[0]
                ) if max_q > 0.0 and np.any(group.mean_q_pred_lpm >= 0.50 * max_q) else np.nan,
                "max_mean_mdot": max_mdot,
            }
        )
    threshold = pd.DataFrame(threshold_rows)
    threshold.to_csv(output_dir / "valve_thresholds.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
    for axis, valve in zip(axes.flat, VALVES):
        for state_id, group in data[data.valve == valve].groupby("state_id"):
            group = group.sort_values("physical_command")
            axis.plot(group.physical_command, group.mean_q_pred_lpm, marker="o", label=f"state {int(state_id)}")
        axis.set_title(valve)
        axis.set_ylabel("mean q_pred [L/min]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("physical command")
    fig.suptitle("lib3 valve effective-opening sweep")
    fig.tight_layout()
    fig.savefig(output_dir / "valve_qpred_sweep.png", dpi=160)
    plt.close(fig)

    print("valve sweep thresholds:")
    print(threshold.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_matched_pid_and_valve_sweep"))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--sweep-points", type=int, default=11)
    parser.add_argument("--warmup-sec", type=float, default=0.5)
    parser.add_argument("--measure-sec", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_pid(args.output_dir, args.episodes, args.steps)
    run_sweep(args.output_dir, args.sweep_points, args.warmup_sec, args.measure_sec)
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
