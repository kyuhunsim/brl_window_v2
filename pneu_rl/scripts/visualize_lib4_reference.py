#!/usr/bin/env python3
"""Create an auditable plot and summary for a lib4 3D trajectory bank.

The image deliberately shows the oracle state and stored command beside the
three RL reference channels.  Oracle commands are diagnostics only; they are
never fed to the policy.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


STATE_NAMES = ("ch_pos", "ch_neg", "act_pos", "act_neg", "contraction_mm", "velocity_mm_s")
VALVE_NAMES = ("ch_pos", "ch_neg", "act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--bank-overview-output",
        type=Path,
        default=None,
        help="optional overview of all pressure/displacement bank members",
    )
    args = parser.parse_args()

    profile = yaml.safe_load(args.reference_profile.read_text(encoding="utf-8"))
    npz_path = Path(profile["kwargs"]["trajectory_npz"])
    if not npz_path.is_absolute():
        npz_path = args.reference_profile.parents[3] / npz_path
    with np.load(npz_path) as data:
        time = np.asarray(data["time"], dtype=np.float64)
        reference = np.asarray(data["reference"], dtype=np.float64)
        state = np.asarray(data["state"], dtype=np.float64)
        raw_action = np.asarray(data["raw_action"], dtype=np.float64)
        physical_action = np.asarray(data["physical_action"], dtype=np.float64)
    if time.ndim == 1:
        time = time[None, :]
    index = int(args.trajectory)
    if not 0 <= index < len(reference):
        raise ValueError(f"trajectory index {index} not in [0, {len(reference)})")
    t, ref, x, raw, physical = (array[index] for array in (time, reference, state, raw_action, physical_action))
    rate = np.gradient(x[:, :5], t, axis=0)
    reserve = np.c_[x[:, 0] - x[:, 2], x[:, 3] - x[:, 1]]
    filters = profile["design"]["filters"]
    block_steps = int(profile["design"]["block_steps"])

    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    axes[0].plot(t, ref[:, 0], "r", label="ACT POS reference")
    axes[0].plot(t, ref[:, 1], "b", label="ACT NEG reference")
    axes[0].set_ylabel("Actuator [kPa]")
    overview_disp_axis = axes[0].twinx()
    overview_disp_axis.plot(
        t,
        ref[:, 2],
        color="tab:green",
        linewidth=1.8,
        label="Contraction reference",
    )
    overview_disp_axis.set_ylabel("Contraction [mm]", color="tab:green")
    overview_disp_axis.tick_params(axis="y", labelcolor="tab:green")
    disp_band = filters["state_bands"]["displacement_mm"]
    overview_disp_axis.set_ylim(float(disp_band[0]), float(disp_band[1]))
    overview_lines = axes[0].get_lines() + overview_disp_axis.get_lines()
    axes[0].legend(
        overview_lines,
        [line.get_label() for line in overview_lines],
        loc="best",
        ncol=3,
        fontsize=8,
    )
    axes[0].set_title(
        "RL reference channels: actuator POS / actuator NEG / contraction"
    )

    axes[1].plot(t, x[:, 0], color="darkred", label="CH POS")
    axes[1].plot(t, x[:, 1], color="navy", label="CH NEG")
    axes[1].plot(t, x[:, 2], "r--", alpha=.75, label="ACT POS")
    axes[1].plot(t, x[:, 3], "b--", alpha=.75, label="ACT NEG")
    axes[1].set_ylabel("All pressure [kPa]")

    axes[2].plot(t, ref[:, 2], color="tab:green", label="Contraction reference")
    axes[2].set_ylabel("Contraction [mm]", color="tab:green")
    axes[2].tick_params(axis="y", labelcolor="tab:green")
    axes[2].set_ylim(float(disp_band[0]), float(disp_band[1]))
    axes[2].axhline(
        float(disp_band[1]),
        color="tab:green",
        linewidth=.9,
        linestyle=":",
        label=f"configured stroke limit ({float(disp_band[1]):g} mm)",
    )
    disp_rate_axis = axes[2].twinx()
    disp_rate_axis.plot(
        t,
        rate[:, 4],
        color="darkgreen",
        alpha=.65,
        label="Contraction rate",
    )
    disp_rate_axis.set_ylabel("Contraction rate [mm/s]", color="darkgreen")
    disp_rate_axis.tick_params(axis="y", labelcolor="darkgreen")
    displacement_lines = axes[2].get_lines() + disp_rate_axis.get_lines()
    axes[2].legend(
        displacement_lines,
        [line.get_label() for line in displacement_lines],
        loc="best",
        ncol=3,
        fontsize=8,
    )

    axes[3].plot(t, reserve[:, 0], color="tab:orange", label="POS reserve: CH POS − ACT POS")
    axes[3].plot(t, reserve[:, 1], color="tab:purple", label="NEG reserve: ACT NEG − CH NEG")
    for value in filters["minimum_reserve_kpa"]:
        axes[3].axhline(value, color="black", linewidth=.8, linestyle=":")
    axes[3].set_ylabel("Reserve [kPa]")

    for valve, values in zip(VALVE_NAMES, raw.T):
        axes[4].plot(t, values, label=f"raw {valve}")
    axes[4].set_ylabel("Oracle raw command")
    axes[4].set_xlabel("Continuous time [s]")
    axes[4].set_ylim(-.03, 1.03)

    for axis in axes:
        for step in range(block_steps, len(t), block_steps):
            axis.axvline(t[step], color="black", linestyle=":", linewidth=.8)
        axis.grid(alpha=.28)
        if axis not in (axes[0], axes[2]):
            axis.legend(loc="best", ncol=2, fontsize=8)
    fig.suptitle(
        "lib4 oracle-feasible 3D reference [ACT POS, ACT NEG, contraction]\n"
        "vertical dotted lines are carry blocks; "
        "stored commands are audit data, not policy observations/actions"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.bank_overview_output is not None:
        overview, overview_axes = plt.subplots(2, 2, figsize=(16, 9))
        for member in range(reference.shape[0]):
            member_time = time[member]
            member_ref = reference[member]
            alpha = 0.9 if member == index else 0.28
            width = 1.8 if member == index else 0.8
            overview_axes[0, 0].plot(member_time, member_ref[:, 0], color="tab:red", alpha=alpha, linewidth=width)
            overview_axes[0, 1].plot(member_time, member_ref[:, 1], color="tab:blue", alpha=alpha, linewidth=width)
            overview_axes[1, 0].plot(member_time, member_ref[:, 2], color="tab:green", alpha=alpha, linewidth=width)
        overview_axes[0, 0].set(title="ACT POS reference bank", ylabel="Pressure [kPa]")
        overview_axes[0, 1].set(title="ACT NEG reference bank", ylabel="Pressure [kPa]")
        overview_axes[1, 0].set(
            title=f"Contraction reference bank (configured stroke 0–{float(disp_band[1]):g} mm)",
            xlabel="Continuous time [s]",
            ylabel="Contraction [mm]",
            ylim=(float(disp_band[0]), float(disp_band[1])),
        )
        flat_ref = reference.reshape(-1, reference.shape[-1])
        flat_velocity = state[:, :, 5].reshape(-1)
        scatter = overview_axes[1, 1].scatter(
            flat_ref[:, 0] - flat_ref[:, 1],
            flat_ref[:, 2],
            c=flat_velocity,
            cmap="coolwarm",
            s=3,
            alpha=.28,
            rasterized=True,
        )
        overview_axes[1, 1].set(
            title="Coupled pressure-difference / contraction coverage",
            xlabel="ACT POS − ACT NEG [kPa]",
            ylabel="Contraction [mm]",
            ylim=(float(disp_band[0]), float(disp_band[1])),
        )
        overview.colorbar(scatter, ax=overview_axes[1, 1], label="Contraction velocity [mm/s]")
        for axis in overview_axes.flat:
            axis.grid(alpha=.28)
        overview.suptitle(
            f"lib4 feasible 3D reference bank — {reference.shape[0]} members; "
            "highlighted curve is the selected trajectory"
        )
        overview.tight_layout()
        args.bank_overview_output.parent.mkdir(parents=True, exist_ok=True)
        overview.savefig(args.bank_overview_output, dpi=180, bbox_inches="tight")
        plt.close(overview)

    summary = {
        "profile": str(args.reference_profile),
        "npz": str(npz_path),
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "trajectory": index,
        "samples": int(len(t)),
        "duration_sec": float(t[-1]),
        "reference_min": ref.min(axis=0).tolist(),
        "reference_max": ref.max(axis=0).tolist(),
        "reference_span": np.ptp(ref, axis=0).tolist(),
        "reference_channels": ["act_pos_kpa", "act_neg_kpa", "contraction_mm"],
        "state_min": x[:, :5].min(axis=0).tolist(),
        "state_max": x[:, :5].max(axis=0).tolist(),
        "max_abs_rate": np.max(np.abs(rate), axis=0).tolist(),
        "reserve_min": reserve.min(axis=0).tolist(),
        "raw_action_min": raw.min(axis=0).tolist(),
        "raw_action_max": raw.max(axis=0).tolist(),
        "physical_action_min": physical.min(axis=0).tolist(),
        "physical_action_max": physical.max(axis=0).tolist(),
        "plot": str(args.output),
        "bank_overview_plot": str(args.bank_overview_output) if args.bank_overview_output else None,
    }
    summary_path = args.summary_output or args.output.with_suffix(".yaml")
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"plot": str(args.output), "summary": str(summary_path), "reference_max": summary["reference_max"]}, sort_keys=False))


if __name__ == "__main__":
    main()
