"""Find pressure targets reachable with one lib3 branch active.

Positive-only diagnostic:
    policy variables: u1, u3, u4
    fixed variables : u2, u5, u6 = physical 0.8

Negative-only diagnostic is the symmetric u2, u5, u6 sweep.  This is an
open-loop feasibility map; it does not train a policy and does not claim that
every point is a good tracking target.  Targets are summarized from the
late-window pressure and rate under each fixed-branch condition.
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
PHYSICAL_LOW = 0.8
PHYSICAL_HIGH = 1.0
STATES = (
    (ATM, ATM, ATM, ATM),
    (140.0, 40.0, 130.0, 70.0),
    (120.0, 20.0, 115.0, 60.0),
    (180.0, 80.0, 145.0, 80.0),
)


def to_raw(physical: float) -> float:
    return float((physical - PHYSICAL_LOW) / (PHYSICAL_HIGH - PHYSICAL_LOW))


def make_sim() -> PneuSim:
    return PneuSim(
        freq=FREQ_HZ,
        delay=0.1,
        noise=False,
        scale=True,
        scale_low=PHYSICAL_LOW,
        scale_high=PHYSICAL_HIGH,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )


def measure(
    sim: PneuSim,
    initial_state: tuple[float, float, float, float],
    action: np.ndarray,
    warmup_steps: int,
    probe_steps: int,
) -> dict[str, float]:
    sim.set_init_press(*initial_state)
    for _ in range(warmup_steps):
        sim.observe(np.zeros(6, dtype=np.float64))

    pressure = []
    debug_rows = []
    for _ in range(probe_steps):
        obs, _ = sim.observe(action)
        pressure.append(obs[1:5].copy())
        debug_rows.append(sim.get_valve_debug())
    pressure = np.asarray(pressure, dtype=float)
    debug = pd.DataFrame(debug_rows)

    late_n = max(2, int(round(0.5 * FREQ_HZ)))
    late = pressure[-late_n:]
    t = np.arange(len(late), dtype=float) * DT
    t -= t.mean()
    denom = float(np.dot(t, t))
    row: dict[str, float] = {}
    names = ("ch_pos", "ch_neg", "act_pos", "act_neg")
    for index, name in enumerate(names):
        values = late[:, index]
        row[f"{name}_late_mean_kpa"] = float(values.mean())
        row[f"{name}_late_min_kpa"] = float(values.min())
        row[f"{name}_late_max_kpa"] = float(values.max())
        row[f"{name}_late_rate_kpa_s"] = float(
            np.dot(t, values - values.mean()) / denom if denom > 0.0 else 0.0
        )

    for valve in ("ch_pos", "ch_neg", "act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
        for quantity in ("z", "q_pred_lpm", "mdot"):
            values = debug[f"{valve}_{quantity}"].to_numpy(dtype=float)
            row[f"{valve}_{quantity}_late_mean"] = float(values[-late_n:].mean())
            row[f"{valve}_{quantity}_late_max_abs"] = float(np.max(np.abs(values[-late_n:])))
    return row


def run_branch(
    branch: str,
    commands: np.ndarray,
    warmup_steps: int,
    probe_steps: int,
) -> pd.DataFrame:
    if branch == "positive":
        active = (0, 2, 3)
        fixed = (1, 4, 5)
        target_name = "act_pos"
    elif branch == "negative":
        active = (1, 4, 5)
        fixed = (0, 2, 3)
        target_name = "act_neg"
    else:
        raise ValueError(branch)

    sim = make_sim()
    rows: list[dict[str, float | str]] = []
    for state_id, initial_state in enumerate(STATES):
        for c0 in commands:
            for c1 in commands:
                for c2 in commands:
                    physical = np.full(6, PHYSICAL_LOW, dtype=float)
                    physical[list(active)] = (c0, c1, c2)
                    action = np.asarray([to_raw(x) for x in physical], dtype=np.float64)
                    result = measure(sim, initial_state, action, warmup_steps, probe_steps)
                    row: dict[str, float | str] = {
                        "branch": branch,
                        "state_id": float(state_id),
                        "u1_physical": float(physical[0]),
                        "u2_physical": float(physical[1]),
                        "u3_physical": float(physical[2]),
                        "u4_physical": float(physical[3]),
                        "u5_physical": float(physical[4]),
                        "u6_physical": float(physical[5]),
                        "fixed_physical": PHYSICAL_LOW,
                        "target_name": target_name,
                    }
                    row.update(result)
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(data: pd.DataFrame, branch: str, output_dir: Path) -> pd.DataFrame:
    target = "act_pos" if branch == "positive" else "act_neg"
    inactive = [x for x in ("ch_pos", "ch_neg", "act_pos", "act_neg") if x != target]
    # A candidate is called "steady-ish" only for target selection; this is not
    # a hard feasibility certificate.  It prevents choosing a point that is
    # still moving rapidly at the end of the open-loop probe.
    rate = data[f"{target}_late_rate_kpa_s"].abs()
    steady = data[rate <= 5.0].copy()
    all_values = data[f"{target}_late_mean_kpa"]
    steady_values = steady[f"{target}_late_mean_kpa"] if len(steady) else all_values
    summary = pd.DataFrame(
        {
            "branch": [branch],
            "all_target_min_kpa": [float(all_values.min())],
            "all_target_p25_kpa": [float(all_values.quantile(0.25))],
            "all_target_median_kpa": [float(all_values.median())],
            "all_target_p75_kpa": [float(all_values.quantile(0.75))],
            "all_target_max_kpa": [float(all_values.max())],
            "steady_candidate_count": [int(len(steady))],
            "steady_target_min_kpa": [float(steady_values.min())],
            "steady_target_p25_kpa": [float(steady_values.quantile(0.25))],
            "steady_target_median_kpa": [float(steady_values.median())],
            "steady_target_p75_kpa": [float(steady_values.quantile(0.75))],
            "steady_target_max_kpa": [float(steady_values.max())],
        }
    )
    summary.to_csv(output_dir / f"{branch}_summary.csv", index=False)

    # Target distribution and target-vs-rate plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(all_values, bins=30, alpha=0.75, label="all grid points")
    if len(steady):
        axes[0].hist(steady_values, bins=30, alpha=0.75, label="|late rate| <= 5")
    axes[0].set_xlabel(f"{target} late mean [kPa]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{branch} fixed-branch reachable pressure")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(all_values, data[f"{target}_late_rate_kpa_s"], s=8, alpha=0.35)
    axes[1].axhline(5.0, color="r", linestyle="--")
    axes[1].axhline(-5.0, color="r", linestyle="--")
    axes[1].set_xlabel(f"{target} late mean [kPa]")
    axes[1].set_ylabel("late rate [kPa/s]")
    axes[1].set_title(f"{branch} target/rate tradeoff")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{branch}_feasibility.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pneu_rl/exp/lib3_fixed_branch_feasibility"))
    parser.add_argument("--commands", default="0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--probe-steps", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = np.asarray([float(x) for x in args.commands.split(",")], dtype=float)

    all_summaries = []
    for branch in ("positive", "negative"):
        data = run_branch(branch, commands, args.warmup_steps, args.probe_steps)
        data.to_csv(args.output_dir / f"{branch}_grid.csv", index=False)
        all_summaries.append(summarize(data, branch, args.output_dir))

    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "physical_scale": [PHYSICAL_LOW, PHYSICAL_HIGH],
        "fixed_controls_physical": PHYSICAL_LOW,
        "commands": commands.tolist(),
        "warmup_steps": args.warmup_steps,
        "probe_steps": args.probe_steps,
        "late_window_sec": 0.5,
        "operating_points": [list(x) for x in STATES],
        "steady_candidate_rule": "absolute late target rate <= 5 kPa/s; target selection still requires closed-loop verification",
    }
    with (args.output_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    print(summary.to_string(index=False))
    print(f"saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
