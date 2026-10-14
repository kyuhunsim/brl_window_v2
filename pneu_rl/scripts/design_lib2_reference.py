"""Characterize lib2 pressure-rate responses and generate a feasible reference.

This script deliberately measures the complete simulator rather than deriving
reference limits from an isolated valve's maximum flow.  The fitted valve
model from ``optimizer.py`` is already embedded in lib2, and the sweep below
therefore includes valve opening dynamics, pump flow, chamber volume, and the
coupling between positive and negative pressure.

Outputs:
  - raw state/action pressure-rate response CSV
  - robust directional pressure-rate summary
  - a ready-to-use feasible-reference YAML profile
  - pressure-rate response and reference-preview figures
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_utils" / "src",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.sim2 import PneuSim
from pneu_ref.feasible_random_ref import FeasibleRandomRef
from pneu_ref.random_ref import RandomRef
from pneu_utils.utils import setup_plot_style


DEFAULT_ANALYSIS_DIR = ROOT / "pneu_rl" / "exp" / "_analysis" / "lib2_reference_design"
DEFAULT_IMAGE_DIR = ROOT / "docs" / "img" / "reference_design"
DEFAULT_PROFILE = ROOT / "pneu_rl" / "config" / "reference" / "lib2_feasible.yaml"


def parse_grid(text: str) -> np.ndarray:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return np.asarray(values, dtype=np.float64)


def robust_directional_rates(
    rows: list[dict[str, float]],
    state_quantile: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Estimate a conservative capacity across the pressure-state grid.

    For every initial pressure pair, find the best action for each direction.
    Then take a low quantile across states.  This avoids using one lucky raw
    maximum as the reference limit.
    """
    grouped: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row["valid"] > 0.5:
            grouped[(row["init_pos_kpa"], row["init_neg_kpa"])].append(row)

    state_rows: list[dict[str, float]] = []
    for (pos, neg), samples in grouped.items():
        pos_rates = np.asarray([item["pos_rate_kpa_s"] for item in samples])
        neg_rates = np.asarray([item["neg_rate_kpa_s"] for item in samples])
        state_rows.append(
            {
                "init_pos_kpa": pos,
                "init_neg_kpa": neg,
                "pos_up": max(float(np.max(pos_rates)), 0.0),
                "pos_down": max(float(-np.min(pos_rates)), 0.0),
                "neg_up": max(float(np.max(neg_rates)), 0.0),
                "neg_down": max(float(-np.min(neg_rates)), 0.0),
            }
        )

    if not state_rows:
        raise RuntimeError("No valid characterization rows were produced")

    capacities = {}
    for key in ("pos_up", "pos_down", "neg_up", "neg_down"):
        values = np.asarray([row[key] for row in state_rows], dtype=np.float64)
        capacities[key] = float(np.quantile(values, state_quantile))
    return capacities, state_rows


def run_sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    sim.set_debug_enabled(True)

    action_grid = np.linspace(0.0, 1.0, args.action_points)
    warmup_steps = max(int(round(args.warmup_sec * args.freq)), 1)
    measure_steps = max(int(round(args.measure_sec * args.freq)), 2)
    rows: list[dict[str, float]] = []

    for init_pos in args.pos_grid:
        for init_neg in args.neg_grid:
            for raw_pos in action_grid:
                for raw_neg in action_grid:
                    sim.set_init_press(float(init_pos), float(init_neg))
                    control = np.array([raw_pos, raw_neg], dtype=np.float64)
                    q_pos: list[float] = []
                    q_neg: list[float] = []

                    valid = True
                    for _ in range(warmup_steps):
                        obs, _ = sim.observe(control)
                        valve = sim.get_valve_debug()
                        q_pos.append(float(valve["pos_q_pred_lpm"]))
                        q_neg.append(float(valve["neg_q_pred_lpm"]))
                        if not np.all(np.isfinite(obs)):
                            valid = False
                            break

                    measure_start = np.asarray(obs[1:3], dtype=np.float64)
                    model_rates: list[tuple[float, float]] = []
                    mdot_net: list[tuple[float, float]] = []
                    for _ in range(measure_steps):
                        obs, _ = sim.observe(control)
                        debug = sim.get_model_debug()
                        valve = sim.get_valve_debug()
                        model_rates.append(
                            (float(debug["dPposdt"]), float(debug["dPnegdt"]))
                        )
                        mdot_net.append(
                            (float(debug["mdot_out"]), float(debug["mdot_in"]))
                        )
                        q_pos.append(float(valve["pos_q_pred_lpm"]))
                        q_neg.append(float(valve["neg_q_pred_lpm"]))
                        if (
                            not np.all(np.isfinite(obs))
                            or obs[1] < args.pressure_floor
                            or obs[1] > args.pressure_ceiling
                            or obs[2] < args.pressure_floor
                            or obs[2] > args.pressure_ceiling
                        ):
                            valid = False
                            break

                    measure_end = np.asarray(obs[1:3], dtype=np.float64)
                    elapsed = max(len(model_rates) / args.freq, 1.0 / args.freq)
                    finite_rate = (measure_end - measure_start) / elapsed
                    model_rate = (
                        np.mean(np.asarray(model_rates), axis=0)
                        if model_rates
                        else np.array([np.nan, np.nan])
                    )
                    net_flow = (
                        np.mean(np.asarray(mdot_net), axis=0)
                        if mdot_net
                        else np.array([np.nan, np.nan])
                    )
                    q_pos_arr = np.asarray(q_pos, dtype=np.float64)
                    q_neg_arr = np.asarray(q_neg, dtype=np.float64)
                    row_valid = bool(
                        valid
                        and len(model_rates) == measure_steps
                        and np.all(np.isfinite(finite_rate))
                    )
                    rows.append(
                        {
                            "init_pos_kpa": float(init_pos),
                            "init_neg_kpa": float(init_neg),
                            "raw_ctrl_pos": float(raw_pos),
                            "raw_ctrl_neg": float(raw_neg),
                            "physical_ctrl_pos": float(
                                args.scale_low
                                + (args.scale_high - args.scale_low) * raw_pos
                            ),
                            "physical_ctrl_neg": float(
                                args.scale_low
                                + (args.scale_high - args.scale_low) * raw_neg
                            ),
                            "measure_start_pos_kpa": float(measure_start[0]),
                            "measure_start_neg_kpa": float(measure_start[1]),
                            "measure_end_pos_kpa": float(measure_end[0]),
                            "measure_end_neg_kpa": float(measure_end[1]),
                            "pos_rate_kpa_s": float(finite_rate[0]),
                            "neg_rate_kpa_s": float(finite_rate[1]),
                            "model_pos_rate_kpa_s": float(model_rate[0]),
                            "model_neg_rate_kpa_s": float(model_rate[1]),
                            "net_pos_mdot_kg_s": float(net_flow[0]),
                            "net_neg_mdot_kg_s": float(net_flow[1]),
                            "pos_valve_q_max_lpm": float(np.nanmax(q_pos_arr)),
                            "neg_valve_q_max_lpm": float(np.nanmax(q_neg_arr)),
                            "pos_valve_q_slew_max_lpm_s": maximum_slew(
                                q_pos_arr, args.freq
                            ),
                            "neg_valve_q_slew_max_lpm_s": maximum_slew(
                                q_neg_arr, args.freq
                            ),
                            "valid": float(row_valid),
                        }
                    )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def maximum_slew(values: np.ndarray, freq: float) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.nanmax(np.abs(np.diff(values))) * freq)


def sample_reference(reference, duration: float, freq: float) -> tuple[np.ndarray, np.ndarray]:
    # PneuRef initializes its buffer at 1/f, so sample the same clock here.
    # Starting legacy RandomRef at exactly t=0 would expose its uninitialized
    # zero fields and create an artificial first-point discontinuity that the
    # environment never uses.
    times = np.arange(1.0 / freq, duration + 0.5 / freq, 1.0 / freq)
    values = np.asarray([reference.get_goal(float(time)) for time in times])
    return times, values


def make_plots(
    rows: list[dict[str, float]],
    state_rows: list[dict[str, float]],
    capacities: dict[str, float],
    safe_rates: dict[str, float],
    profile_kwargs: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    setup_plot_style(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    args.image_dir.mkdir(parents=True, exist_ok=True)

    valid = [row for row in rows if row["valid"] > 0.5]
    pos_rate = np.asarray([row["pos_rate_kpa_s"] for row in valid])
    neg_rate = np.asarray([row["neg_rate_kpa_s"] for row in valid])
    ctrl_pos = np.asarray([row["physical_ctrl_pos"] for row in valid])
    ctrl_neg = np.asarray([row["physical_ctrl_neg"] for row in valid])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    scatter = axes[0].scatter(
        pos_rate,
        neg_rate,
        c=ctrl_pos - ctrl_neg,
        cmap="coolwarm",
        alpha=0.7,
        s=24,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"$\dot P_{pos}$ [kPa/s]")
    axes[0].set_ylabel(r"$\dot P_{neg}$ [kPa/s]")
    axes[0].set_title("Sampled joint pressure-rate response")
    fig.colorbar(scatter, ax=axes[0], label=r"$u_{pos}-u_{neg}$")

    labels = ["POS up", "POS down", "NEG up", "NEG down"]
    keys = ["pos_up", "pos_down", "neg_up", "neg_down"]
    x = np.arange(len(keys))
    axes[1].bar(
        x - 0.18,
        [capacities[key] for key in keys],
        width=0.36,
        color="gray",
        label=f"robust Q{args.state_quantile:.2f}",
    )
    axes[1].bar(
        x + 0.18,
        [safe_rates[key] for key in keys],
        width=0.36,
        color=["red", "firebrick", "blue", "navy"],
        label=f"reference ({args.safety_factor:.2f}×)",
    )
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Pressure-rate magnitude [kPa/s]")
    axes[1].set_title("Directional capacity used for reference")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.image_dir / "lib2_pressure_rate_reachability.png", dpi=180)
    plt.close(fig)

    # The scatter alone does not show which valve action produced a point.
    # Add a representative-state action map so the POS-valve sign and the
    # cross-channel coupling can be inspected directly.
    median_pos = float(np.median([row["init_pos_kpa"] for row in valid]))
    median_neg = float(np.median([row["init_neg_kpa"] for row in valid]))
    representative = min(
        valid,
        key=lambda row: (
            row["init_pos_kpa"] - median_pos
        ) ** 2
        + (row["init_neg_kpa"] - median_neg) ** 2,
    )
    representative_rows = [
        row
        for row in valid
        if row["init_pos_kpa"] == representative["init_pos_kpa"]
        and row["init_neg_kpa"] == representative["init_neg_kpa"]
    ]
    action_pos = np.asarray(
        sorted({row["physical_ctrl_pos"] for row in representative_rows})
    )
    action_neg = np.asarray(
        sorted({row["physical_ctrl_neg"] for row in representative_rows})
    )
    rate_maps = []
    for key in ("pos_rate_kpa_s", "neg_rate_kpa_s"):
        rate_map = np.full((len(action_pos), len(action_neg)), np.nan)
        for row in representative_rows:
            i = int(np.argmin(np.abs(action_pos - row["physical_ctrl_pos"])))
            j = int(np.argmin(np.abs(action_neg - row["physical_ctrl_neg"])))
            rate_map[i, j] = row[key]
        rate_maps.append(rate_map)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    for axis, rate_map, channel in zip(
        axes,
        rate_maps,
        ("POS", "NEG"),
    ):
        limit = float(np.nanmax(np.abs(rate_map)))
        image = axis.imshow(
            rate_map,
            origin="lower",
            aspect="auto",
            extent=(action_neg[0], action_neg[-1], action_pos[0], action_pos[-1]),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.contour(
            action_neg,
            action_pos,
            rate_map,
            levels=[0.0],
            colors="black",
            linewidths=1.0,
        )
        axis.set_title(
            rf"$\dot P_{{{channel.lower()}}}$ at "
            rf"$(P_{{pos}},P_{{neg}})=({representative['init_pos_kpa']:.1f},"
            rf"{representative['init_neg_kpa']:.1f})$"
        )
        axis.set_xlabel("Physical $u_{neg}$")
        axis.set_ylabel("Physical $u_{pos}$")
        fig.colorbar(image, ax=axis, label="Pressure rate [kPa/s]")
    fig.tight_layout()
    fig.savefig(args.image_dir / "lib2_action_rate_heatmap.png", dpi=180)
    plt.close(fig)

    state_pos = np.asarray([row["init_pos_kpa"] for row in state_rows])
    state_neg = np.asarray([row["init_neg_kpa"] for row in state_rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for axis, key, title in zip(
        axes.ravel(),
        ("pos_up", "pos_down", "neg_up", "neg_down"),
        ("POS up", "POS down", "NEG up", "NEG down"),
    ):
        values = np.asarray([row[key] for row in state_rows])
        plot = axis.scatter(state_pos, state_neg, c=values, cmap="viridis", s=110)
        axis.set_title(title)
        axis.set_xlabel("Initial POS [kPa]")
        axis.set_ylabel("Initial NEG [kPa]")
        fig.colorbar(plot, ax=axis, label="Best rate [kPa/s]")
    fig.tight_layout()
    fig.savefig(args.image_dir / "lib2_rate_capacity_by_pressure.png", dpi=180)
    plt.close(fig)

    legacy = RandomRef(
        seed=args.ref_seed,
        pos_max_ts=5,
        neg_max_ts=5,
        pos_max_amp=20,
        neg_max_amp=5,
        pos_max_per=10,
        neg_max_per=10,
        pos_min_off=140,
        pos_max_off=240,
        neg_min_off=10,
        neg_max_off=32,
    )
    feasible = FeasibleRandomRef(**profile_kwargs)
    time, legacy_values = sample_reference(legacy, args.preview_sec, args.freq)
    _, feasible_values = sample_reference(feasible, args.preview_sec, args.freq)
    legacy_slew = np.r_[
        np.zeros((1, 2)),
        np.abs(np.diff(legacy_values, axis=0)) * args.freq,
    ]
    feasible_slew = np.r_[
        np.zeros((1, 2)),
        np.abs(np.diff(feasible_values, axis=0)) * args.freq,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, (name, values, slew) in enumerate(
        (
            ("Legacy RandomRef", legacy_values, legacy_slew),
            ("FeasibleRandomRef", feasible_values, feasible_slew),
        )
    ):
        axes[0, col].plot(time, values[:, 0], color="red", label="POS")
        axes[0, col].plot(time, values[:, 1], color="blue", label="NEG")
        axes[0, col].set_title(name)
        axes[0, col].set_ylabel("Reference pressure [kPa]")
        axes[0, col].legend()
        axes[1, col].plot(time, slew[:, 0], color="red", label="POS")
        axes[1, col].plot(time, slew[:, 1], color="blue", label="NEG")
        axes[1, col].set_xlabel("Time [s]")
        axes[1, col].set_ylabel(r"$|\dot P^{ref}|$ [kPa/s]")
        axes[1, col].legend()
    fig.tight_layout()
    fig.savefig(args.image_dir / "legacy_vs_feasible_reference.png", dpi=180)
    plt.close(fig)

    return {
        "legacy_pos_slew_max_kpa_s": float(np.max(legacy_slew[:, 0])),
        "legacy_neg_slew_max_kpa_s": float(np.max(legacy_slew[:, 1])),
        "feasible_pos_slew_max_kpa_s": float(np.max(feasible_slew[:, 0])),
        "feasible_neg_slew_max_kpa_s": float(np.max(feasible_slew[:, 1])),
        "representative_state_pos_kpa": float(representative["init_pos_kpa"]),
        "representative_state_neg_kpa": float(representative["init_neg_kpa"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--scale-low", type=float, default=0.8)
    parser.add_argument("--scale-high", type=float, default=1.0)
    parser.add_argument("--pos-grid", type=parse_grid, default=parse_grid("140,165,190,215,240"))
    parser.add_argument("--neg-grid", type=parse_grid, default=parse_grid("10,17.5,25,32"))
    parser.add_argument("--action-points", type=int, default=7)
    parser.add_argument("--warmup-sec", type=float, default=0.35)
    parser.add_argument("--measure-sec", type=float, default=0.25)
    parser.add_argument("--pressure-floor", type=float, default=1.0)
    parser.add_argument("--pressure-ceiling", type=float, default=450.0)
    parser.add_argument("--state-quantile", type=float, default=0.2)
    parser.add_argument("--safety-factor", type=float, default=0.5)
    parser.add_argument(
        "--valve-wn-min-rad-s",
        type=float,
        default=11.0,
        help="minimum compiled lib2 valve natural frequency",
    )
    parser.add_argument(
        "--reference-bandwidth-fraction",
        type=float,
        default=0.1,
        help="reference frequency / minimum valve natural frequency",
    )
    parser.add_argument("--profile-pos-min", type=float, default=140.0)
    parser.add_argument("--profile-pos-max", type=float, default=220.0)
    parser.add_argument("--profile-neg-min", type=float, default=15.0)
    parser.add_argument("--profile-neg-max", type=float, default=30.0)
    parser.add_argument("--initial-pos", type=float, default=101.325)
    parser.add_argument("--initial-neg", type=float, default=101.325)
    parser.add_argument("--initial-hold", type=float, default=1.0)
    parser.add_argument("--hold-min", type=float, default=1.5)
    parser.add_argument("--hold-max", type=float, default=4.0)
    parser.add_argument("--min-transition", type=float, default=2.0)
    parser.add_argument("--ref-seed", type=int, default=61098)
    parser.add_argument("--preview-sec", type=float, default=40.96)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()

    if not 0.0 <= args.state_quantile <= 1.0:
        parser.error("--state-quantile must be in [0, 1]")
    if not 0.0 < args.safety_factor <= 1.0:
        parser.error("--safety-factor must be in (0, 1]")
    if args.action_points < 2:
        parser.error("--action-points must be at least 2")

    rows = run_sweep(args)
    capacities, state_rows = robust_directional_rates(rows, args.state_quantile)
    uncontrollable = [key for key, value in capacities.items() if value <= 0.0]
    if uncontrollable:
        raise RuntimeError(
            "No robust controllability in these directions: "
            f"{uncontrollable}. Narrow the pressure range or change the "
            "physical action range before generating a reference profile."
        )
    safe_rates = {
        key: value * args.safety_factor for key, value in capacities.items()
    }

    valve_natural_hz = args.valve_wn_min_rad_s / (2.0 * math.pi)
    reference_frequency_hz = (
        valve_natural_hz * args.reference_bandwidth_fraction
    )
    profile_kwargs = {
        "seed": args.ref_seed,
        "pos_min": args.profile_pos_min,
        "pos_max": args.profile_pos_max,
        "neg_min": args.profile_neg_min,
        "neg_max": args.profile_neg_max,
        "pos_rate_up": safe_rates["pos_up"],
        "pos_rate_down": safe_rates["pos_down"],
        "neg_rate_up": safe_rates["neg_up"],
        "neg_rate_down": safe_rates["neg_down"],
        "hold_min": args.hold_min,
        "hold_max": args.hold_max,
        "min_transition": args.min_transition,
        "max_frequency_hz": reference_frequency_hz,
        "initial_pos": args.initial_pos,
        "initial_neg": args.initial_neg,
        "initial_hold": args.initial_hold,
        "min_pos_delta": 5.0,
        "min_neg_delta": 2.0,
    }
    profile = {
        "kind": "feasible_random",
        "kwargs": profile_kwargs,
        "design": {
            "library": 2,
            "action_scale": [args.scale_low, args.scale_high],
            "state_quantile": args.state_quantile,
            "pressure_rate_safety_factor": args.safety_factor,
            "valve_wn_min_rad_s": args.valve_wn_min_rad_s,
            "valve_natural_frequency_min_hz": valve_natural_hz,
            "reference_bandwidth_fraction": args.reference_bandwidth_fraction,
            "source": "lib2 full-plant state/action sweep",
        },
    }

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.profile_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_dir / "reachability.csv", rows)
    write_csv(args.analysis_dir / "state_directional_capacity.csv", state_rows)
    with args.profile_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)

    preview_metrics = make_plots(
        rows,
        state_rows,
        capacities,
        safe_rates,
        profile_kwargs,
        args,
    )
    valid = [row for row in rows if row["valid"] > 0.5]
    summary = {
        "sweep": {
            "pressure_states": len(state_rows),
            "action_pairs_per_state": args.action_points**2,
            "valid_samples": len(valid),
            "total_samples": len(rows),
            "warmup_sec": args.warmup_sec,
            "measure_sec": args.measure_sec,
        },
        "robust_directional_capacity_kpa_s": capacities,
        "safe_reference_rate_kpa_s": safe_rates,
        "maximum_observed": {
            "net_pos_mdot_abs_kg_s": float(
                np.max(np.abs([row["net_pos_mdot_kg_s"] for row in valid]))
            ),
            "net_neg_mdot_abs_kg_s": float(
                np.max(np.abs([row["net_neg_mdot_kg_s"] for row in valid]))
            ),
            "pos_valve_flow_lpm": float(
                np.max([row["pos_valve_q_max_lpm"] for row in valid])
            ),
            "neg_valve_flow_lpm": float(
                np.max([row["neg_valve_q_max_lpm"] for row in valid])
            ),
            "pos_valve_flow_slew_lpm_s": float(
                np.max([row["pos_valve_q_slew_max_lpm_s"] for row in valid])
            ),
            "neg_valve_flow_slew_lpm_s": float(
                np.max([row["neg_valve_q_slew_max_lpm_s"] for row in valid])
            ),
        },
        "frequency_design": {
            "valve_wn_min_rad_s": args.valve_wn_min_rad_s,
            "valve_natural_frequency_min_hz": valve_natural_hz,
            "reference_frequency_hz": reference_frequency_hz,
        },
        "preview": preview_metrics,
        "profile": str(args.profile_path),
    }
    with (args.analysis_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    print(f"[ INFO] profile: {args.profile_path}")
    print(f"[ INFO] analysis: {args.analysis_dir}")
    print(f"[ INFO] figures: {args.image_dir}")
    print(f"[ INFO] safe rates [kPa/s]: {safe_rates}")


if __name__ == "__main__":
    main()
