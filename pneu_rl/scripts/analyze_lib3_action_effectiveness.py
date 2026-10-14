"""Measure lib3 valve action effectiveness around representative operating points.

This is a diagnostic, not a training/evaluation script.  For each operating
point, all valves are first held at the physical lower command (0.8).  One
valve is then stepped to 0.80--1.00 while the other valves remain at 0.8.
The measured pressure-rate change is compared with the all-0.8 baseline.

The script also measures equal-command inlet/outlet pairs to expose actuator
conflict/cancellation and records the hidden valve quantities exposed by the
lib3 simulator: z, effective area, q_pred, and mdot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim3 import PneuSim


ATM = 101.325
FREQ_HZ = 50.0
DT = 1.0 / FREQ_HZ
VALVES = (
    "ch_pos",
    "ch_neg",
    "act_pos_in",
    "act_pos_out",
    "act_neg_in",
    "act_neg_out",
)
OUTPUTS = ("ch_pos", "ch_neg", "act_pos", "act_neg")
STATES = (
    (ATM, ATM, ATM, ATM),
    (140.0, 40.0, 130.0, 70.0),
    (120.0, 20.0, 115.0, 60.0),
    (180.0, 80.0, 145.0, 80.0),
)


def physical_to_normalized(value: float, low: float = 0.8, high: float = 1.0) -> float:
    return float((value - low) / (high - low))


def make_sim() -> PneuSim:
    return PneuSim(
        freq=FREQ_HZ,
        delay=0.1,
        noise=False,
        scale=True,
        scale_low=0.8,
        scale_high=1.0,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )


def pressure_dict(obs: np.ndarray) -> dict[str, float]:
    # PneuSim returns [time, chamber_pos, chamber_neg, act_pos, act_neg].
    return {
        "ch_pos": float(obs[1]),
        "ch_neg": float(obs[2]),
        "act_pos": float(obs[3]),
        "act_neg": float(obs[4]),
    }


def collect_segment(
    sim: PneuSim,
    initial_state: tuple[float, float, float, float],
    action: np.ndarray,
    warmup_steps: int,
    probe_steps: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    sim.set_init_press(*initial_state)
    baseline_action = np.zeros(6, dtype=np.float64)
    for _ in range(warmup_steps):
        sim.observe(baseline_action)

    rows: list[dict[str, float]] = []
    for step in range(probe_steps):
        obs, _ = sim.observe(action)
        pressure = pressure_dict(obs)
        debug = sim.get_valve_debug()
        row: dict[str, float] = {"step": float(step), "time": float((step + 1) * DT)}
        row.update(pressure)
        for key, value in debug.items():
            row[key] = float(value)
        rows.append(row)
    data = pd.DataFrame(rows)

    # Least-squares slope is less sensitive to simulator quantization than a
    # single finite difference.  Windows are intentionally reported both early
    # and late because the valve dynamics are not instantaneous.
    summaries: dict[str, float] = {}
    windows = {
        "early_0p2": slice(0, max(2, int(round(0.2 * FREQ_HZ)))),
        "early_0p5": slice(0, max(2, int(round(0.5 * FREQ_HZ)))),
        "late_0p2": slice(max(0, probe_steps - int(round(0.2 * FREQ_HZ))), probe_steps),
    }
    for window_name, window in windows.items():
        part = data.iloc[window]
        t = part["time"].to_numpy(dtype=float)
        t = t - t.mean()
        denom = float(np.dot(t, t))
        for output in OUTPUTS:
            y = part[output].to_numpy(dtype=float)
            slope = float(np.dot(t, y - y.mean()) / denom) if denom > 0.0 else 0.0
            summaries[f"rate_{output}_{window_name}_kpa_s"] = slope

    for valve in VALVES:
        for name in ("z", "area_eff", "q_pred_lpm", "mdot"):
            values = data[f"{valve}_{name}"].to_numpy(dtype=float)
            summaries[f"{valve}_{name}_last"] = float(values[-1])
            summaries[f"{valve}_{name}_mean"] = float(np.mean(values))
            summaries[f"{valve}_{name}_max_abs"] = float(np.max(np.abs(values)))
    return data, summaries


def make_action(values: dict[int, float]) -> np.ndarray:
    action = np.zeros(6, dtype=np.float64)
    for index, physical in values.items():
        action[index] = physical_to_normalized(physical)
    return action


def run_single_valve_sweep(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    commands = np.array([float(x) for x in args.commands.split(",")], dtype=float)
    sim = make_sim()
    rows: list[dict[str, float | str]] = []

    for state_id, initial_state in enumerate(STATES):
        _, baseline = collect_segment(
            sim,
            initial_state,
            make_action({}),
            args.warmup_steps,
            args.probe_steps,
        )
        for valve_index, valve in enumerate(VALVES):
            for command in commands:
                _, result = collect_segment(
                    sim,
                    initial_state,
                    make_action({valve_index: command}),
                    args.warmup_steps,
                    args.probe_steps,
                )
                row: dict[str, float | str] = {
                    "state_id": float(state_id),
                    "valve_index": float(valve_index),
                    "valve": valve,
                    "initial_ch_pos_kpa": float(initial_state[0]),
                    "initial_ch_neg_kpa": float(initial_state[1]),
                    "initial_act_pos_kpa": float(initial_state[2]),
                    "initial_act_neg_kpa": float(initial_state[3]),
                    "physical_command": float(command),
                }
                for key, value in result.items():
                    row[key] = value
                    if key.startswith("rate_"):
                        row[f"delta_{key}"] = value - baseline[key]
                rows.append(row)

    data = pd.DataFrame(rows)
    data.to_csv(output_dir / "single_valve_action_map.csv", index=False)
    return data


def run_pair_conflict_sweep(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    commands = np.array([float(x) for x in args.pair_commands.split(",")], dtype=float)
    pairs = ((2, 3, "act_pos_pair"), (4, 5, "act_neg_pair"))
    sim = make_sim()
    rows: list[dict[str, float | str]] = []

    for state_id, initial_state in enumerate(STATES):
        _, baseline = collect_segment(
            sim,
            initial_state,
            make_action({}),
            args.warmup_steps,
            args.probe_steps,
        )
        for inlet_index, outlet_index, pair_name in pairs:
            for inlet_command in commands:
                for outlet_command in commands:
                    _, result = collect_segment(
                        sim,
                        initial_state,
                        make_action({inlet_index: inlet_command, outlet_index: outlet_command}),
                        args.warmup_steps,
                        args.probe_steps,
                    )
                    row: dict[str, float | str] = {
                        "state_id": float(state_id),
                        "pair": pair_name,
                        "inlet_command": float(inlet_command),
                        "outlet_command": float(outlet_command),
                    }
                    for key in result:
                        if key.startswith("rate_"):
                            row[key] = result[key]
                            row[f"delta_{key}"] = result[key] - baseline[key]
                    for valve_index in (inlet_index, outlet_index):
                        valve = VALVES[valve_index]
                        for name in ("z", "q_pred_lpm", "mdot"):
                            row[f"{valve}_{name}_last"] = result[f"{valve}_{name}_last"]
                    rows.append(row)

    data = pd.DataFrame(rows)
    data.to_csv(output_dir / "inlet_outlet_conflict_map.csv", index=False)
    return data


def plot_single_map(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=True)
    for axis, valve in zip(axes.flat, VALVES):
        group = data[data["valve"] == valve]
        for state_id, state_group in group.groupby("state_id"):
            state_group = state_group.sort_values("physical_command")
            axis.plot(
                state_group["physical_command"],
                state_group["delta_rate_act_pos_early_0p2_kpa_s"],
                marker="o",
                label=f"state {int(state_id)}",
            )
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_title(f"{valve}: Δ dP_act_pos/dt")
        axis.set_ylabel("Δ rate [kPa/s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("physical command")
    fig.suptitle("lib3 action-effectiveness map: actuator POS early response")
    fig.tight_layout()
    fig.savefig(output_dir / "action_map_act_pos_rate.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for axis, output in zip(axes.flat, OUTPUTS):
        for valve in VALVES:
            group = data[data["valve"] == valve]
            means = group.groupby("physical_command")[f"delta_rate_{output}_early_0p2_kpa_s"].mean()
            axis.plot(means.index, means.values, marker="o", label=valve)
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_title(f"Δ dP_{output}/dt")
        axis.set_ylabel("Δ rate [kPa/s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("physical command")
    fig.suptitle("lib3 action map averaged over operating points")
    fig.tight_layout()
    fig.savefig(output_dir / "action_map_all_outputs.png", dpi=160)
    plt.close(fig)


def plot_conflict(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, pair in zip(axes, ("act_pos_pair", "act_neg_pair")):
        group = data[data["pair"] == pair]
        value_name = "delta_rate_act_pos_early_0p2_kpa_s" if pair == "act_pos_pair" else "delta_rate_act_neg_early_0p2_kpa_s"
        pivot = group.groupby(["inlet_command", "outlet_command"])[value_name].mean().unstack()
        image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
            cmap="coolwarm",
        )
        axis.set_title(f"{pair}: Δ early actuator rate")
        axis.set_xlabel("outlet command")
        axis.set_ylabel("inlet command")
        fig.colorbar(image, ax=axis, label="Δ rate [kPa/s]")
    fig.suptitle("lib3 inlet/outlet simultaneous-command conflict map")
    fig.savefig(output_dir / "inlet_outlet_conflict_map.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pneu_rl/exp/lib3_action_effectiveness_map"),
    )
    parser.add_argument("--commands", default="0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--pair-commands", default="0.80,0.90,0.95,1.00")
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--probe-steps", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single = run_single_valve_sweep(args, args.output_dir)
    conflict = run_pair_conflict_sweep(args, args.output_dir)
    plot_single_map(single, args.output_dir)
    plot_conflict(conflict, args.output_dir)

    metadata = {
        "lib": 3,
        "freq_hz": FREQ_HZ,
        "sim_delay_sec": 0.1,
        "scale_low": 0.8,
        "scale_high": 1.0,
        "operating_points": [list(state) for state in STATES],
        "single_valve_commands": [float(x) for x in args.commands.split(",")],
        "pair_commands": [float(x) for x in args.pair_commands.split(",")],
        "warmup_steps": args.warmup_steps,
        "probe_steps": args.probe_steps,
        "interpretation": "rates are finite-difference simulator outputs relative to all-0.8 baseline; early windows expose delay/transient response",
    }
    with (args.output_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    summary = (
        single.groupby("valve")
        .agg(
            max_abs_delta_act_pos_rate=("delta_rate_act_pos_early_0p2_kpa_s", lambda x: float(np.max(np.abs(x)))),
            max_abs_delta_act_neg_rate=("delta_rate_act_neg_early_0p2_kpa_s", lambda x: float(np.max(np.abs(x)))),
            max_q_pred=("act_pos_in_q_pred_lpm_last", "max"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "single_valve_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
