"""Audit native, simultaneous lib3 states and flows recorded during policy rollouts.

No fitting, sensor correction, pressure clipping, or model modification occurs here.
The delayed policy observation is deliberately not used for pressure/flow tests.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

ATM = 101.325
STD_RHO = 1.20411831637462
PTR = ctypes.POINTER(ctypes.c_double)
PRESSURE_NAMES = ("ch_pos", "ch_neg", "act_pos", "act_neg")
GAS_NAMES = ("ch_pos", "ch_neg", "pump_cylinder_1", "pump_cylinder_2", "act_pos", "act_neg")
VALVE_NAMES = ("ch_pos", "ch_neg", "act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out")
DEBUG_NAMES = ("u_eff", "current", "state_curr", "h", "force_net", "area_eff", "q_static_lpm", "q_pred_lpm", "mdot")


class NativeStateSampler:
    """Read a PneuSim C ABI without stepping, resetting, or altering its state."""

    def __init__(self, observation):
        self.lib = getattr(observation, "lib", observation)
        self.lib.get_augmented_state_dim_c.restype = ctypes.c_int
        self.dimension = int(self.lib.get_augmented_state_dim_c())
        self.lib.get_augmented_state_schema_c.restype = ctypes.c_int
        self.schema = int(self.lib.get_augmented_state_schema_c())
        if self.schema != 2 or self.dimension != 37:
            raise ValueError(f"Expected opening-state lib3 schema 2/dimension 37, got {self.schema}/{self.dimension}")
        self.lib.get_augmented_state_c.argtypes = [PTR]
        self.lib.get_valve_debug_c.restype = PTR
        self.lib.get_mass_flowrate_c.restype = PTR

    def sample(self):
        state = np.empty(self.dimension, dtype=np.float64)
        self.lib.get_augmented_state_c(state.ctypes.data_as(PTR))
        debug = np.ctypeslib.as_array(self.lib.get_valve_debug_c(), shape=(54,)).copy().reshape(6, 9)
        flow = np.ctypeslib.as_array(self.lib.get_mass_flowrate_c(), shape=(10,)).copy()
        return {"native_state": state, "valve_debug": debug,
                "mass_flowrate": flow, "runtime_state_schema": self.schema}


def valve_pressures(state):
    """Absolute kPa at the six modelled one-way solenoid valve ports."""
    p = np.asarray(state)[..., [0, 1, 5, 6]]
    ambient = np.full(p.shape[:-1], ATM)
    inlet = np.stack((p[..., 0], ambient, p[..., 0], p[..., 2], ambient, p[..., 3]), axis=-1)
    outlet = np.stack((ambient, p[..., 1], p[..., 2], ambient, p[..., 3], p[..., 1]), axis=-1)
    return inlet, outlet


def compressible_phi(inlet, outlet):
    """The source model's dimensionless ideal-gas orifice factor (K=1.4)."""
    gamma = 1.4
    ratio = np.clip(np.maximum(outlet, 0.) / np.maximum(inlet, 1e-9), 0., 1.)
    critical = (2. / (gamma + 1.)) ** (gamma / (gamma - 1.))
    choked = np.sqrt(gamma * (2. / (gamma + 1.)) ** ((gamma + 1.) / (gamma - 1.)))
    subsonic = np.sqrt(2. * gamma / (gamma - 1.)) * np.sqrt(np.maximum(
        ratio ** (2. / gamma) - ratio ** ((gamma + 1.) / gamma), 0.))
    return np.where(ratio <= critical, choked, subsonic)


def _maximum(values):
    return float(np.max(values)) if np.size(values) else 0.


def _counts(mask, names):
    return {name: int(np.asarray(mask)[..., i].sum()) for i, name in enumerate(names)}


def _range(values, names):
    values = np.asarray(values)
    flat = values.reshape(-1, values.shape[-1])
    return {name: {"min": float(flat[:, i].min()), "max": float(flat[:, i].max())}
            for i, name in enumerate(names)}


def analyze_arrays(arrays):
    state = np.asarray(arrays["native_state"], dtype=float)
    debug = np.asarray(arrays["valve_debug"], dtype=float)
    control = np.asarray(arrays["physical_command"], dtype=float)
    if state.ndim != 3 or state.shape[-1] != 37:
        raise ValueError(f"native_state needs (members,steps,37), got {state.shape}")
    leading = state.shape[:2]
    if debug.shape != leading + (6, 9) or control.shape != leading + (6,):
        raise ValueError("Native state, valve debug, and physical command arrays must align")
    if "runtime_state_schema" in arrays:
        if not np.all(np.asarray(arrays["runtime_state_schema"]) == 2):
            raise ValueError("The rollout includes a non-opening runtime state schema")
    gas = state[..., [0, 1, 3, 4, 5, 6]]
    pressure = state[..., [0, 1, 5, 6]]
    runtime = state[..., 7:].reshape(leading + (6, 5))
    h, opening, target = runtime[..., 0], runtime[..., 1], runtime[..., 2]
    inlet, outlet = valve_pressures(state)
    head = inlet - outlet
    q = debug[..., 7]
    mdot = debug[..., 8]
    area = debug[..., 5]
    q_recomputed = area * inlet * compressible_phi(inlet, outlet)
    conversion_error = mdot - q * STD_RHO / 60000.
    current_expected = .30 * np.clip((np.clip(control, 0., 1.) - .5) * 2., 0., 1.)
    wrong_head = (head < -1e-7) & (q > 1e-6)
    violations = {
        "nonfinite_native_state": int((~np.isfinite(state)).sum()),
        "nonfinite_valve_debug": int((~np.isfinite(debug)).sum()),
        "nonfinite_physical_command": int((~np.isfinite(control)).sum()),
        "nonpositive_absolute_gas_pressure": int((gas <= 0.).sum()),
        "h_outside_minus1_plus1": int((np.abs(h) > 1. + 1e-10).sum()),
        "opening_fraction_outside_0_1": int(((opening < -1e-12) | (opening > 1. + 1e-12)).sum()),
        "target_opening_outside_0_1": int(((target < -1e-12) | (target > 1. + 1e-12)).sum()),
        "negative_area": int((area < -1e-12).sum()),
        "negative_solenoid_flow": int((q < -1e-10).sum()),
        "nonzero_flow_against_negative_head": int(wrong_head.sum()),
        "physical_command_outside_0p8_1": int(((control < .8 - 1e-10) | (control > 1. + 1e-10)).sum()),
        "flow_area_pressure_formula_mismatch": int((np.abs(q - q_recomputed) > 1e-8).sum()),
        "standard_lpm_mass_conversion_mismatch": int((np.abs(conversion_error) > 1e-14).sum()),
        "debug_runtime_h_mismatch": int((np.abs(debug[..., 3] - h) > 1e-10).sum()),
        "runtime_current_debug_mismatch": int((np.abs(runtime[..., 3] - debug[..., 1]) > 1e-10).sum()),
        "runtime_branch_outside_0_1": int(((runtime[..., 4] < -1e-12) | (runtime[..., 4] > 1. + 1e-12)).sum()),
        "current_physical_command_mismatch": int((np.abs(debug[..., 1] - current_expected) > 1e-10).sum()),
    }
    report = {
        "passed": not any(violations.values()), "members": leading[0], "steps_per_member": leading[1],
        "sampled_control_states": int(np.prod(leading)), "sampled_valve_states": int(np.prod(leading) * 6),
        "violations": violations, "all_six_gas_pressure_kpa": _range(gas, GAS_NAMES),
        "four_observed_port_pressures_kpa": _range(pressure, PRESSURE_NAMES),
        "normalized_hysteresis": _range(h, VALVE_NAMES), "opening_fraction": _range(opening, VALVE_NAMES),
        "target_opening_fraction": _range(target, VALVE_NAMES),
        "physical_command": _range(control, VALVE_NAMES), "valve_flow_standard_lpm": _range(q, VALVE_NAMES),
        "head_kpa": _range(head, VALVE_NAMES),
        "wrong_head_flow_by_valve": _counts(wrong_head, VALVE_NAMES),
        "maximum_wrong_head_flow_lpm": _maximum(q[wrong_head]),
        "max_abs_flow_area_formula_error_lpm": _maximum(np.abs(q - q_recomputed)),
        "max_abs_standard_flow_mass_conversion_error_kg_s": _maximum(np.abs(conversion_error)),
        "max_abs_current_command_error_ampere": _maximum(np.abs(debug[..., 1] - current_expected)),
        "operating_envelope_separate_from_physical_invariants": {
            "bounds_kpa": {"ch_pos": [ATM, 220.], "ch_neg": [10., ATM],
                           "act_pos": [ATM, None], "act_neg": [0., ATM]},
            "out_of_bounds_sample_counts": {
                "ch_pos": int(((pressure[..., 0] < ATM - 1e-7) | (pressure[..., 0] > 220. + 1e-7)).sum()),
                "ch_neg": int(((pressure[..., 1] < 10. - 1e-7) | (pressure[..., 1] > ATM + 1e-7)).sum()),
                "act_pos_below_atmosphere": int((pressure[..., 2] < ATM - 1e-7).sum()),
                "act_neg_above_atmosphere": int((pressure[..., 3] > ATM + 1e-7).sum()),
            },
            "ch_pos_above_220_kpa_fraction": float(np.mean(pressure[..., 0] > 220. + 1e-7)),
            "ch_pos_maximum_excess_above_220_kpa": max(0., float(pressure[..., 0].max()) - 220.),
            "ch_pos_above_220_samples_per_member": (pressure[..., 0] > 220. + 1e-7).sum(axis=1).tolist(),
            "positive_supply_reserve_kpa": _range((pressure[..., 0] - pressure[..., 2])[..., None], ["ch_pos_minus_act_pos"]),
            "negative_supply_reserve_kpa": _range((pressure[..., 3] - pressure[..., 1])[..., None], ["act_neg_minus_ch_neg"]),
            "low_supply_reserve_fractions": {
                "positive": {f"below_{threshold}kpa": float(np.mean(pressure[..., 0] - pressure[..., 2] < threshold))
                             for threshold in (0, 5, 15)},
                "negative": {f"below_{threshold}kpa": float(np.mean(pressure[..., 3] - pressure[..., 1] < threshold))
                             for threshold in (0, 5, 15)},
            },
        },
        "control_at_bounds_fraction": {
            name: {"at_0p8": float(np.mean(control[..., i] <= .8 + 1e-5)),
                   "at_1": float(np.mean(control[..., i] >= 1. - 1e-5)),
                   "within_1pct_action_span_of_0p8": float(np.mean(control[..., i] <= .802)),
                   "within_1pct_action_span_of_1": float(np.mean(control[..., i] >= .998))}
            for i, name in enumerate(VALVE_NAMES)
        },
        "interpretation": [
            "These checks establish physical/algebraic feasibility on the sampled closed-loop trajectories only.",
            "They do not establish real-world accuracy, global stability, or positivity between stored samples.",
            "Pressure and flow are native simultaneous C ABI outputs; delayed policy observations are not used for direction checks.",
            "One-way valve assumptions are those already present in this model, not a proof of actual hardware check-valve behavior.",
            "No mass-balance residual is inferred from 50 Hz finite differences of pressure; unresolved pump dynamics would invalidate that test.",
            "Command saturation and a loss of supply reserve are reported separately and are not by themselves physical violations.",
            "The 0.8 command limit is not a claim of complete valve closure; residual opening/flow at that command is not automatically a feasibility violation.",
        ],
    }
    if "mass_flowrate" in arrays:
        mass = np.asarray(arrays["mass_flowrate"], dtype=float)
        if mass.shape != leading + (10,):
            raise ValueError("mass_flowrate needs (members,steps,10)")
        pump_inlet = np.stack((state[..., 3], state[..., 1], state[..., 4], state[..., 1]), axis=-1)
        pump_outlet = np.stack((state[..., 0], state[..., 3], state[..., 0], state[..., 4]), axis=-1)
        pump_wrong = ((pump_inlet - pump_outlet) < -1e-7) & (mass[..., :4] > 1e-12)
        violations.update({
            "nonfinite_mass_flowrate": int((~np.isfinite(mass)).sum()),
            "negative_pump_or_solenoid_mass_flow": int((mass < -1e-14).sum()),
            "pump_check_valve_flow_against_head": int(pump_wrong.sum()),
            "mass_flow_debug_index_mismatch": int((np.abs(mass[..., 4:] - mdot) > 1e-14).sum()),
        })
        report["mass_flow_checks"] = {
            "pump_wrong_head_counts": _counts(pump_wrong, ["pump1_out", "pump1_in", "pump2_out", "pump2_in"]),
            "max_abs_debug_mass_flow_difference_kg_s": _maximum(np.abs(mass[..., 4:] - mdot)),
            "maximum_pump_wrong_head_flow_kg_s": _maximum(mass[..., :4][pump_wrong]),
        }
        report["passed"] = not any(violations.values())
    return report


def write_plots(arrays, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    state = arrays["native_state"]
    runtime = state[..., 7:].reshape(state.shape[:2] + (6, 5))
    pressure = state[..., [0, 1, 5, 6]]
    inlet, outlet = valve_pressures(state)
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for i, name in enumerate(VALVE_NAMES):
        ax = axes.flat[i]
        # Deterministic subsampling keeps the artifact compact without affecting statistics.
        head = (inlet - outlet)[..., i].ravel()[::8]
        flow = arrays["valve_debug"][..., i, 7].ravel()[::8]
        opening = runtime[..., i, 1].ravel()[::8]
        scatter = ax.scatter(head, flow, c=opening, s=3, vmin=0, vmax=1, cmap="viridis", alpha=.6, rasterized=True)
        ax.axvline(0, color="gray", lw=.8)
        ax.set(title=name, xlabel="Instantaneous head (kPa)", ylabel="Standard flow (L/min)")
        ax.grid(alpha=.2)
    figure.colorbar(scatter, ax=list(axes.flat), label="Opening fraction", shrink=.8)
    figure.suptitle("Closed-loop feasibility: flow versus simultaneous pressure head")
    figure.savefig(directory / "flow_head_feasibility.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].boxplot([pressure[..., i].ravel() for i in range(4)], tick_labels=PRESSURE_NAMES, showfliers=False)
    axes[0].set(ylabel="Absolute pressure (kPa)", title="Pressure distribution")
    axes[1].boxplot([runtime[..., i, 1].ravel() for i in range(6)], tick_labels=np.arange(1, 7), showfliers=False)
    axes[1].axhline(0, color="gray", lw=.8); axes[1].axhline(1, color="gray", lw=.8)
    axes[1].set(xlabel="Valve index", ylabel="Opening fraction", title="Opening state domain")
    axes[2].boxplot([runtime[..., i, 0].ravel() for i in range(6)], tick_labels=np.arange(1, 7), showfliers=False)
    axes[2].axhline(-1, color="gray", lw=.8); axes[2].axhline(1, color="gray", lw=.8)
    axes[2].set(xlabel="Valve index", ylabel="Normalized hysteresis h", title="Hysteresis state domain")
    figure.savefig(directory / "state_domain.png", dpi=150)
    plt.close(figure)


def analyze_rollout(path, output_dir, plots=True):
    path, output_dir = Path(path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    report = analyze_arrays(arrays)
    report.update(rollout_path=str(path.resolve()), rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output_dir / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    if plots:
        write_plots(arrays, output_dir)
    lines = [f"# Closed-loop feasibility: {path.parent.name}", "",
             f"Result: {'PASS' if report['passed'] else 'FAIL'} on {report['members']} members, "
             f"{report['sampled_control_states']:,} control states, {report['sampled_valve_states']:,} valve states.", "",
             "| Invariant | Violation count |", "| --- | ---: |"]
    lines += [f"| {key} | {value} |" for key, value in report["violations"].items()]
    lines += ["", "| Absolute gas pressure | Minimum (kPa) | Maximum (kPa) |", "| --- | ---: | ---: |"]
    lines += [f"| {key} | {value['min']:.6f} | {value['max']:.6f} |" for key, value in report["all_six_gas_pressure_kpa"].items()]
    lines += ["", "The operating envelope and command saturation are reported separately in feasibility.json.", ""]
    lines += [f"- {text}" for text in report["interpretation"]]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = analyze_rollout(args.rollout, args.output_dir, plots=not args.no_plots)
    print(json.dumps({"passed": report["passed"], "samples": report["sampled_control_states"], "violations": report["violations"]}))


if __name__ == "__main__":
    main()
