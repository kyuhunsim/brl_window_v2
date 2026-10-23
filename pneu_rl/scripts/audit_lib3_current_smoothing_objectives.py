"""Audit stored lib3 rollouts without executing a simulator or learning.

Hypothetical action penalties are evaluated on fixed trajectories only. Their
magnitude cannot predict the policy, tracking error, or vibration after training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW = ROOT / "pneu_rl/exp/lib3_physical_training_review_20261023"
VALVES = ("ch_pos", "ch_neg", "act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delay_diff(array, initial):
    return np.diff(array, axis=1, prepend=initial[:, None, ...])


def consecutive_mask(mask, steps):
    result = np.zeros_like(mask, dtype=bool)
    run = np.zeros(mask.shape[0], dtype=int)
    for i in range(mask.shape[1]):
        run = np.where(mask[:, i], run + 1, 0)
        result[:, i] = run >= steps
    return result


def mean_at(array, mask):
    return np.mean(array[mask], axis=0)


def spectral_summary(array, mask, frequency):
    """PSD per independent member: no artificial joins between trajectories."""
    spectra = []
    for member in range(array.shape[0]):
        x = array[member, mask[member]]
        if len(x) < 128:
            continue
        f, psd = welch(x, fs=frequency, axis=0,
                       nperseg=min(1024, len(x)), detrend="linear")
        df = f[1] - f[0]
        total = np.sum(psd[(f >= .1) & (f <= 20)], axis=0) * df
        high = np.sum(psd[(f >= 2) & (f <= 20)], axis=0) * df
        spectra.append((total, high))
    if not spectra:
        return None
    total = np.mean([item[0] for item in spectra], axis=0)
    high = np.mean([item[1] for item in spectra], axis=0)
    return dict(rms_01_20_hz=np.sqrt(total).tolist(),
                rms_2_20_hz=np.sqrt(high).tolist(),
                fraction_2_20_in_01_20_hz=np.divide(
                    high, total, out=np.zeros_like(high), where=total > 1e-24).tolist())


def analyze(model_dir, mode, review):
    folder = review / model_dir.name
    if mode == "constant_holds":
        folder /= "constant_holds"
    paths = [model_dir / "cfg.yaml", folder / "rollout.npz", folder / "metadata.json"]
    cfg = yaml.safe_load(paths[0].read_text())
    metadata = json.loads(paths[2].read_text())
    with np.load(paths[1]) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    frequency = metadata["frequency_hz"]
    w = cfg["env"]["rwd_kwargs"]
    assert w["tracking_loss_mode"] == "abs"
    assert all(w.get(name, 0.) == 0. for name in (
        "pos_prev_rwd_coeff", "neg_prev_rwd_coeff", "pos_fut_rwd_coeff",
        "neg_fut_rwd_coeff", "pos_pred_rwd_coeff", "neg_pred_rwd_coeff",
        "action_delta_rwd_coeff", "second_action_delta_rwd_coeff",
        "conflict_rwd_coeff", "chamber_reserve_rwd_coeff"))
    p = data["pressure_observed"]
    r = data["reference"]
    initial_pressure = np.asarray([row["pressure"] for row in metadata["initial_states"]])
    # PneuRef.reset pre-fills the reference buffer with get_goal(1/f), not
    # measured pressure. Its first 10 current references retain that value.
    # The constant-hold diagnostic explicitly replaces it with start pressure.
    initial_reference = initial_pressure[:, 2:] if mode == "constant_holds" else r[:, 0]
    if mode != "constant_holds":
        np.testing.assert_array_equal(r[:, :10], np.repeat(r[:, :1], 10, axis=1))
    dp = delay_diff(p[:, :, 2:], initial_pressure[:, 2:]) * frequency
    dr = delay_diff(r, initial_reference) * frequency
    error = r - p[:, :, 2:]
    tracking = np.abs(error) * [w["pos_curr_rwd_coeff"], w["neg_curr_rwd_coeff"]]
    rate = np.abs(dr - dp) * [
        w["pos_rate_rwd_coeff"] / w["pos_rate_scale_kpa_s"],
        w["neg_rate_rwd_coeff"] / w["neg_rate_scale_kpa_s"]]
    bound = w["chamber_bound_rwd_coeff"] * (
        np.maximum(w["chamber_pos_min_kpa"] - p[:, :, 0], 0.)
        + np.maximum(p[:, :, 0] - w["chamber_pos_max_kpa"], 0.)
        + np.maximum(w["chamber_neg_min_kpa"] - p[:, :, 1], 0.)
        + np.maximum(p[:, :, 1] - w["chamber_neg_max_kpa"], 0.))
    reconstructed_reward = -tracking.sum(-1) - rate.sum(-1) - bound
    reconstruction_error = float(np.max(np.abs(reconstructed_reward-data["reward"])))
    np.testing.assert_allclose(reconstructed_reward, data["reward"], rtol=0., atol=1e-9)

    low, high = cfg["obs"]["scale_low"], cfg["obs"]["scale_high"]
    u = (data["physical_command"] - low) / (high - low)
    initial_action = np.asarray([row["action_history"][-1] for row in metadata["initial_states"]])
    du = delay_diff(u, initial_action)
    ddu = delay_diff(du, np.zeros_like(initial_action))
    # Runtime schema 2: each valve stores [h, opening, target, previous_u, previous_p].
    opening = data["native_state"][:, :, 7:].reshape(*p.shape[:2], 6, 5)[:, :, :, 1]
    flow = data["valve_debug"][:, :, :, 7]
    time_mask = np.broadcast_to(data["time"] > 5., p.shape[:2])
    after10 = np.broadcast_to(data["time"] > 10., p.shape[:2])
    channel_plateau = np.stack([
        consecutive_mask(np.abs(dr[:, :, k]) < .2, int(.5*frequency)) & time_mask
        for k in range(2)], axis=-1)
    masks = dict(whole=np.ones(p.shape[:2], dtype=bool), after5=time_mask,
                 after10=after10, both_channels_plateau=channel_plateau.all(-1))
    rows = []
    for scope, mask in masks.items():
        if not mask.any():
            continue
        tr = mean_at(tracking, mask)
        ra = mean_at(rate, mask)
        total = float(np.mean(-data["reward"][mask]))
        mse_du = float(np.mean(du[mask]**2))
        mse_ddu = float(np.mean(ddu[mask]**2))
        row = dict(model=model_dir.name, type=metadata["calibration_type"],
                   policy=metadata["policy"], mode=mode, scope=scope,
                   samples=int(mask.sum()),
                   tracking_cost_pos=float(tr[0]), tracking_cost_neg=float(tr[1]),
                   rate_cost_pos=float(ra[0]), rate_cost_neg=float(ra[1]),
                   rate_to_tracking_pos=float(ra[0]/max(tr[0], 1e-15)),
                   rate_to_tracking_neg=float(ra[1]/max(tr[1], 1e-15)),
                   mean_total_cost=total, mean_bound_cost=float(np.mean(bound[mask])),
                   tracking_mae_pos=float(np.mean(np.abs(error[mask, 0]))),
                   tracking_mae_neg=float(np.mean(np.abs(error[mask, 1]))),
                   mean_error_rate_abs_pos=float(np.mean(np.abs((dr-dp)[mask, 0]))),
                   mean_error_rate_abs_neg=float(np.mean(np.abs((dr-dp)[mask, 1]))),
                   action_delta_mse_normalized=mse_du,
                   second_action_delta_mse_normalized=mse_ddu,
                   lambda_action_delta_equal_to_10pct_current_cost=.1*total/max(mse_du, 1e-15),
                   lambda_second_delta_equal_to_10pct_current_cost=.1*total/max(mse_ddu, 1e-15))
        for weight in (.01, .1, .5, 1., 5.):
            key = str(weight).replace(".", "p")
            row[f"posthoc_delta_lambda_{key}_fraction_current_cost"] = weight*mse_du/max(total, 1e-15)
            row[f"posthoc_second_delta_lambda_{key}_fraction_current_cost"] = weight*mse_ddu/max(total, 1e-15)
        rows.append(row)

    late = after10 if mode == "constant_holds" else time_mask
    # Delta comparison excludes initial handover; units remain explicit.
    valid_delta = late.copy()
    valid_delta[:, 0] = False
    changes = dict(
        command_normalized_delta_rms=np.sqrt(mean_at(du**2, valid_delta)).tolist(),
        opening_fraction_delta_rms=np.sqrt(mean_at(np.diff(opening, axis=1, prepend=opening[:, :1])**2,
                                                 valid_delta)).tolist(),
        flow_lpm_delta_rms=np.sqrt(mean_at(np.diff(flow, axis=1, prepend=flow[:, :1])**2,
                                         valid_delta)).tolist(),
        command_normalized=spectral_summary(u, late, frequency),
        opening_fraction=spectral_summary(opening, late, frequency),
        flow_lpm=spectral_summary(flow, late, frequency),
        comparison_scope="after10 seconds for constant holds; after5 seconds for bank; native opening/flow vs issued command, not delayed sensor values")
    plateau_rows = []
    for k, channel in enumerate(("pos", "neg")):
        mask = channel_plateau[:, :, k]
        plateau_rows.append(dict(channel=channel, samples=int(mask.sum()),
            tracking_cost=float(np.mean(tracking[:, :, k][mask])),
            rate_cost=float(np.mean(rate[:, :, k][mask]))))
    result = dict(model=model_dir.name, type=metadata["calibration_type"],
                  policy=metadata["policy"], mode=mode,
                  maximum_reward_reconstruction_error=reconstruction_error,
                  inputs_sha256={str(path.relative_to(ROOT)): sha256(path) for path in paths},
                  weights=w, scopes=rows, channel_plateau=plateau_rows,
                  command_opening_flow=changes)
    return result


def plot(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cases = [r for r in results if r["mode"] == "constant_holds"]
    labels = [f'{r["type"]} {r["policy"]}' for r in cases]
    x = np.arange(len(cases))
    rows = [next(row for row in r["scopes"] if row["scope"] == "after10") for r in cases]
    for k, channel in enumerate(("pos", "neg")):
        axes[0].bar(x+(k-.5)*.35, [r[f"rate_to_tracking_{channel}"]*100 for r in rows],
                    width=.35, label=channel.upper())
    axes[0].set_ylabel("Rate penalty / pressure tracking penalty [%]")
    axes[0].set_title("Existing objective: constant holds after10s")
    axes[0].legend()
    axes[1].bar(x-.18, [r["action_delta_mse_normalized"] for r in rows], .36, label="First difference")
    axes[1].bar(x+.18, [r["second_action_delta_mse_normalized"] for r in rows], .36, label="Second difference")
    axes[1].set_ylabel("Mean squared difference, normalized command [0,1]")
    axes[1].set_title("Candidate penalty magnitudes, no learning")
    axes[1].legend()
    for r in cases:
        case = r["command_opening_flow"]
        i = cases.index(r)
        axes[2].bar(i-.18, np.mean(case["command_normalized_delta_rms"]), .36,
                    color="tab:blue", label="Command [0,1]" if i == 0 else None)
        axes[2].bar(i+.18, np.mean(case["opening_fraction_delta_rms"]), .36,
                    color="tab:orange", label="Opening [0,1]" if i == 0 else None)
    axes[2].set_ylabel("Per-step delta RMS, mean over 6 valves")
    axes[2].set_title("Command changes and physical opening differ")
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "current_objective_and_action_changes.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.review / "smoothing_review"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in (f"261023_lib3_opening_{kind}_{policy}_smooth_seed0"
                 for kind in ("type1", "type2") for policy in ("H", "HR")):
        for mode in ("training_bank", "constant_holds"):
            results.append(analyze(ROOT / "pneu_rl/models" / name, mode, args.review))
    report = dict(scope="Offline arithmetic and spectra from existing deterministic rollouts only; no simulator rollout, training, fine-tuning or policy/config edits.",
                  limitations=[
                      "Posthoc action penalties describe scale on fixed data only; they do not estimate learning improvement or tracking/smoothness tradeoffs.",
                      "Normalized command [0,1], physical opening [0,1], and flow [L/min] are distinct quantities; absolute spectral amplitudes with different units are not comparable.",
                      "A plateau uses abs(reference backward difference)<0.2 kPa/s for >=0.5s after5s. Constant-hold spectra use after10s.",
                      "Sinusoidal error derivation ignores startup, sensor delay effects and bias; it explains objective weighting, not the cause of oscillation."],
                  sinusoidal_rate_to_tracking_ratio_per_hz=dict(pos=2*np.pi*.025/(12*.3), neg=2*np.pi*.025/(6*.3)),
                  results=results)
    (output / "current_objective_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    rows = [row for result in results for row in result["scopes"]]
    with (output / "current_objective_scopes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(results, output)
    print(json.dumps(dict(output=str(output), cases=len(results),
                         max_reward_reconstruction_error=max(r["maximum_reward_reconstruction_error"] for r in results)), indent=2))


if __name__ == "__main__":
    main()
