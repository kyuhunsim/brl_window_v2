#!/usr/bin/env python3
"""Compare lib3 observability when candidate state variables are measured.

This is a companion to ``analyze_lib3_nonlinear_observability.py``.  It keeps
the same 25-dimensional continuous perturbation, action sequences, delay
shift, central finite differences, and effective-rank criterion, but changes
the output map.  The added valve/pump variables are simulator-privileged
measurements, not claims about real sensors.

The command itself is not a plant state.  A fixed known command sequence has
zero derivative with respect to the initial plant state, so appending ``u`` to
the output map cannot increase local observability rank.  Previous commands
are nevertheless useful policy information because they condition the hybrid
valve branch; that is discussed in the generated report rather than counted
as a new sensor output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_lib3_nonlinear_observability as base


DEFAULT_OUTPUT = base.ROOT / "pneu_rl/exp/lib3_observability_observation_sets"
MAX_WINDOW_DEFAULT = 20


def valve_indices(component: int) -> list[int]:
    return [7 + valve * base.VALVE_RUNTIME_DIM + component for valve in range(base.VALVE_COUNT)]


OUTPUT_SETS = {
    "pressure_4": [0, 1, 5, 6],
    "pressure_pump_phase": [0, 1, 5, 6, 2, 3, 4],
    "pressure_valve_z": [0, 1, 5, 6] + valve_indices(0),
    "pressure_valve_q": [0, 1, 5, 6] + valve_indices(1),
    "pressure_valve_qdot": [0, 1, 5, 6] + valve_indices(2),
    "pressure_valve_flow_pair": [
        0,
        1,
        5,
        6,
        *valve_indices(1),
        *valve_indices(2),
    ],
    "pressure_all_continuous": [
        *range(7),
        *valve_indices(0),
        *valve_indices(1),
        *valve_indices(2),
    ],
}

# This is a diagnostic control-input augmentation.  The six command rows are
# appended as known inputs, so their derivative with respect to the initial
# plant state is exactly zero.  It should therefore have the same rank as
# ``pressure_4`` (up to numerical noise), unlike the privileged state sets.
COMMAND_OUTPUT_SET = "pressure_plus_current_action"

OUTPUT_LABELS = {
    "pressure_4": "4 pressures",
    "pressure_pump_phase": "pressures + pump + phase",
    "pressure_valve_z": "pressures + valve hysteresis z",
    "pressure_valve_q": "pressures + valve flow q",
    "pressure_valve_qdot": "pressures + valve flow rate q_dot",
    "pressure_valve_flow_pair": "pressures + q + q_dot",
    "pressure_all_continuous": "25 smooth-core states",
    COMMAND_OUTPUT_SET: "4 pressures + current 6-command input",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--delay-steps", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--prepare-steps", type=int, default=10)
    parser.add_argument("--fd-fraction", type=float, default=2e-4)
    parser.add_argument(
        "--rank-absolute-tol",
        type=float,
        default=1e-7,
        help="fixed tolerance after state/output normalization",
    )
    return parser.parse_args()


def rollout_states(
    probe: base.Lib3Probe,
    snapshot: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    probe.restore(snapshot)
    rows = [probe.snapshot()]
    for action in actions:
        probe.step(action)
        rows.append(probe.snapshot())
    return np.asarray(rows, dtype=np.float64)


def finite_difference_state_trajectory(
    probe: base.Lib3Probe,
    snapshot: np.ndarray,
    actions: np.ndarray,
    scales: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = rollout_states(probe, snapshot, actions)
    jacobian = np.empty(
        (baseline.size, len(base.CONTINUOUS_INDICES)), dtype=np.float64
    )
    for column, (state_index, scale) in enumerate(
        zip(base.CONTINUOUS_INDICES, scales)
    ):
        delta = max(float(scale) * fraction, 1e-8)
        plus = snapshot.copy()
        minus = snapshot.copy()
        plus[state_index] += delta
        minus[state_index] -= delta
        y_plus = rollout_states(probe, plus, actions)
        y_minus = rollout_states(probe, minus, actions)
        jacobian[:, column] = ((y_plus - y_minus) / (2.0 * delta)).reshape(-1)
    probe.restore(snapshot)
    return baseline, jacobian


def feature_scale_lookup(
    snapshot: np.ndarray,
    preparation: np.ndarray,
    state_scales: np.ndarray,
) -> dict[int, float]:
    lookup = dict(zip(base.CONTINUOUS_INDICES.tolist(), state_scales.tolist()))
    # The output-only feature scales use the same characteristic units as the
    # perturbed state.  This makes a directly exposed z/q/q_dot coordinate
    # numerically comparable to its corresponding hidden state direction.
    for index in range(base.STATE_DIM):
        if index not in lookup:
            name = base.STATE_NAMES[index]
            if name.endswith("I_prev"):
                lookup[index] = 0.1
            elif name.endswith("branch"):
                lookup[index] = 1.0
            else:
                lookup[index] = max(1.0, 0.1 * abs(float(snapshot[index])))
    return lookup


def select_rows(
    full_jacobian: np.ndarray,
    total_samples: int,
    delay: int,
    window: int,
    feature_indices: list[int],
) -> np.ndarray:
    tensor = full_jacobian.reshape(total_samples, base.STATE_DIM, -1)
    return tensor[delay : delay + window, feature_indices, :].reshape(
        -1, tensor.shape[-1]
    )


def scaled_rank(
    jacobian: np.ndarray,
    state_scales: np.ndarray,
    output_scales: np.ndarray,
    absolute_tol: float,
) -> tuple[int, np.ndarray]:
    scaled = (jacobian * state_scales[None, :]) / output_scales[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0, singular
    rank = int(np.sum(singular > absolute_tol))
    return rank, singular


def configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    library = args.output_dir / "libpneumatic_simulator_observability.so"
    base.compile_probe(library)
    probe = base.Lib3Probe(library)
    operating_state, preparation, final_action = base.prepare_operating_state(
        probe, args.prepare_steps
    )
    state_scales = base.continuous_scales(operating_state, preparation)
    state_scale_lookup = feature_scale_lookup(
        operating_state, preparation, state_scales
    )
    max_window = max(args.windows)
    action_count = max(args.delay_steps) + max_window - 1
    scenarios = {
        "constant": base.plateau_actions(action_count, final_action),
        "time_varying": base.multisine_actions(action_count, 1.17),
    }
    summary_rows: list[dict[str, object]] = []
    spectra: dict[tuple[str, str, int], np.ndarray] = {}

    for scenario, actions in scenarios.items():
        _, full_jacobian = finite_difference_state_trajectory(
            probe,
            operating_state,
            actions,
            state_scales,
            args.fd_fraction,
        )
        total_samples = action_count + 1
        for set_name in [*OUTPUT_SETS.keys(), COMMAND_OUTPUT_SET]:
            feature_indices = (
                OUTPUT_SETS["pressure_4"]
                if set_name == COMMAND_OUTPUT_SET
                else OUTPUT_SETS[set_name]
            )
            if set_name == COMMAND_OUTPUT_SET:
                pressure_indices = OUTPUT_SETS["pressure_4"]
                pressure_scales = np.asarray(
                    [state_scale_lookup[index] for index in pressure_indices],
                    dtype=np.float64,
                )
                feature_scales = np.r_[pressure_scales, [1.0] * 6]
            else:
                feature_scales = np.asarray(
                    [state_scale_lookup[index] for index in feature_indices],
                    dtype=np.float64,
                )
            for delay in args.delay_steps:
                for window in args.windows:
                    if set_name == COMMAND_OUTPUT_SET:
                        pressure_rows = select_rows(
                            full_jacobian,
                            total_samples,
                            delay,
                            window,
                            np.asarray(pressure_indices, dtype=np.int64),
                        )
                        # The appended current-action output is known and
                        # independent of the perturbed initial state.
                        command_rows = np.zeros(
                            (window * 6, len(base.CONTINUOUS_INDICES)),
                            dtype=np.float64,
                        )
                        jacobian = np.vstack((pressure_rows, command_rows))
                    else:
                        jacobian = select_rows(
                            full_jacobian,
                            total_samples,
                            delay,
                            window,
                            feature_indices,
                        )
                    output_scales = (
                        np.r_[np.tile(pressure_scales, window), np.ones(6 * window)]
                        if set_name == COMMAND_OUTPUT_SET
                        else np.tile(feature_scales, window)
                    )
                    rank, singular = scaled_rank(
                        jacobian,
                        state_scales,
                        output_scales,
                        args.rank_absolute_tol,
                    )
                    spectra[(scenario, set_name, delay)] = singular
                    summary_rows.append(
                        {
                            "scenario": scenario,
                            "observation_set": set_name,
                            "observation_label": OUTPUT_LABELS[set_name],
                            "delay_steps": delay,
                            "window_samples": window,
                            "output_dimension_per_step": len(feature_indices)
                            + (6 if set_name == COMMAND_OUTPUT_SET else 0),
                            "output_rows": jacobian.shape[0],
                            "continuous_states": len(base.CONTINUOUS_INDICES),
                            "effective_rank": rank,
                            "sigma_max": float(singular[0]) if len(singular) else 0.0,
                            "sigma_min": float(singular[-1]) if len(singular) else 0.0,
                        }
                    )

    summary_path = args.output_dir / "observation_set_rank_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    configure_plotting()
    plot_sets = [
        "pressure_4",
        "pressure_pump_phase",
        "pressure_valve_z",
        "pressure_valve_q",
        "pressure_valve_qdot",
        "pressure_valve_flow_pair",
        "pressure_all_continuous",
    ]
    # Reserve a dedicated third column for the colorbar.  Using
    # ``figure.colorbar(..., ax=axes)`` lets Matplotlib steal space from the
    # right heatmap, which makes the old figure overlap the panel title.
    figure = plt.figure(figsize=(16.0, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        wspace=0.28,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1], sharey=None),
    ]
    colorbar_axis = figure.add_subplot(grid[0, 2])
    plot_labels = [
        "4 pressures",
        "pressure + pump + phase",
        "pressure + hysteresis z",
        "pressure + valve flow q",
        "pressure + valve flow rate q_dot",
        "pressure + q + q_dot",
        "25 smooth-core states",
    ]
    image = None
    for axis, scenario in zip(axes, ("constant", "time_varying")):
        matrix = []
        for set_name in plot_sets:
            row = []
            for window in args.windows:
                item = next(
                    item
                    for item in summary_rows
                    if item["scenario"] == scenario
                    and item["observation_set"] == set_name
                    and item["delay_steps"] == max(args.delay_steps)
                    and item["window_samples"] == window
                )
                row.append(int(item["effective_rank"]))
            matrix.append(row)
        image = axis.imshow(
            np.asarray(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=len(base.CONTINUOUS_INDICES),
        )
        axis.set_title(
            "Constant input" if scenario == "constant" else "Time-varying input",
            pad=12,
        )
        axis.set_xlabel("Output-history samples")
        axis.set_xticks(range(len(args.windows)), args.windows)
        axis.set_yticks(range(len(plot_sets)), plot_labels)
        axis.tick_params(axis="y", labelsize=9)
        axis.set_xticks(
            np.arange(-0.5, len(args.windows), 1), minor=True
        )
        axis.set_yticks(np.arange(-0.5, len(plot_sets), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8, alpha=0.75)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(plot_sets)):
            for column in range(len(args.windows)):
                axis.text(
                    column,
                    row,
                    str(matrix[row][column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row][column] < 14 else "black",
                    fontsize=10,
                    fontweight="medium",
                )
    axes[0].set_ylabel("Observation set", labelpad=10)
    axes[1].tick_params(axis="y", labelleft=False)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Effective local rank", labelpad=10)
    figure.suptitle(
        "lib3 local effective rank by observation set (0.1 s measurement delay)",
        fontsize=17,
    )
    rank_plot = args.output_dir / "observation_set_rank_heatmap.png"
    figure.savefig(rank_plot, dpi=220, bbox_inches="tight")
    plt.close(figure)

    spectrum_sets = [
        "pressure_4",
        "pressure_valve_z",
        "pressure_valve_q",
        "pressure_valve_qdot",
        "pressure_valve_flow_pair",
        "pressure_all_continuous",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for axis, scenario in zip(axes, ("constant", "time_varying")):
        for set_name in spectrum_sets:
            singular = spectra[(scenario, set_name, max(args.delay_steps))]
            axis.semilogy(
                np.arange(1, len(singular) + 1),
                singular,
                marker="o",
                markersize=2.8,
                linewidth=1.4,
                label=OUTPUT_LABELS[set_name],
            )
        axis.set_title(
            "Constant input" if scenario == "constant" else "Time-varying input"
        )
        axis.set_xlabel("Singular-value index")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="lower left")
    axes[0].set_ylabel("Scaled singular value")
    figure.suptitle("lib3 observation-set singular spectra at H20 and 0.1 s delay", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    spectrum_plot = args.output_dir / "observation_set_singular_spectrum_h20.png"
    figure.savefig(spectrum_plot, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "state_dimension_continuous": len(base.CONTINUOUS_INDICES),
        "output_sets": {
            name: {
                "label": OUTPUT_LABELS[name],
                "state_indices": indices,
            }
            for name, indices in OUTPUT_SETS.items()
        },
        "command_output_set": {
            "name": COMMAND_OUTPUT_SET,
            "label": OUTPUT_LABELS[COMMAND_OUTPUT_SET],
            "note": "Six current-command rows have zero derivative with respect to initial plant state; they are a known-input diagnostic, not a state measurement.",
        },
        "delay_steps": args.delay_steps,
        "windows": args.windows,
        "rank_absolute_tolerance_after_scaling": args.rank_absolute_tol,
        "note": "Valve z/q/q_dot outputs are simulator-privileged measurements; action is a known input, not a state output.",
    }
    with (args.output_dir / "observation_set_analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved: {summary_path}")
    print(f"saved: {rank_plot}")
    print(f"saved: {spectrum_plot}")


if __name__ == "__main__":
    main()
