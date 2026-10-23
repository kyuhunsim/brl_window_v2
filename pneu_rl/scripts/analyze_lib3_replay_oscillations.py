#!/usr/bin/env python3
"""Read-only spectral audit of recorded lib3 replays under fixed command holds.

This audits replay CSVs, without fitting or modifying simulator parameters.  The
50 Hz recordings cannot identify physical oscillations above the Nyquist limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram


CHANNELS = ("press_pos", "press_neg", "act_pos_press", "act_neg_press")
BANDS = ((.05, .5), (.5, 3.), (3., 20.))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(values, fs):
    return periodogram(values, fs=fs, detrend="linear", scaling="density")


def energy(freq, psd, low, high):
    ix = (freq >= low) & (freq <= high)
    # Periodogram bin powers are PSD * df. Rectangular integration also retains
    # the sole low-frequency bin in short fixed-control holds.
    return float(np.sum(psd[ix]) * (freq[1]-freq[0])) if len(freq) > 1 else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-root", type=Path, default=Path("../pneu_tune/exp/lib3_pump_feasible_20261022"))
    p.add_argument("--output-dir", type=Path, default=Path("../pneu_tune/exp/lib3_physical_model_20261023"))
    p.add_argument("--variant", default="baseline")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, parameter_rows, hashes, datasets, spectrum_examples = [], [], {}, {}, {}
    fs = 50.
    for kind in ("type1", "type2"):
        config_path = args.replay_root / kind / "config.json"
        config = json.loads(config_path.read_text())
        data_path = Path(config["data"][0]["path"])
        replay_path = args.replay_root / kind / "replay_1.csv"
        raw = pd.read_csv(data_path).drop_duplicates("time", keep="last")
        replay = pd.read_csv(replay_path)
        hashes.update({str(x): digest(x) for x in (config_path, data_path, replay_path)})
        times, commands = raw.time.to_numpy(), raw[[f"ctrl{v}" for v in range(1, 7)]].to_numpy()
        starts = np.flatnonzero(np.r_[True, np.max(np.abs(np.diff(commands, axis=0)), axis=1) > 1e-6])
        ends = np.r_[starts[1:], len(times)]
        for start, end in zip(starts, ends):
            low, high = times[start], times[end - 1]
            if high - low < 5:
                continue
            if np.max(np.ptp(commands[start:end], axis=0)) > 1e-6:
                continue
            # Retain transient and tail versions separately. A fixed command does
            # not guarantee the pneumatic plant is at steady state.
            for scope, trim in (("after_1s", 1.), ("tail", min(5., (high-low)/3))):
                subset = replay[(replay.source_time >= low + trim) & (replay.source_time <= high)]
                if len(subset) < 150:
                    continue
                for channel in CHANNELS:
                    for model in ("real", "sim"):
                        values = subset[f"{model}_{channel}"].to_numpy()
                        freq, psd = spectrum(values, fs)
                        ix = (freq >= .05) & (freq <= 20.)
                        row = dict(type=kind, scope=scope, start_sec=float(low), end_sec=float(high),
                                   analyzed_start_sec=float(low+trim), n=len(values), channel=channel,
                                   model=model, minimum_kpa=float(values.min()), maximum_kpa=float(values.max()),
                                   range_kpa=float(np.ptp(values)),
                                   peak_hz=float(freq[ix][np.argmax(psd[ix])]),
                                   rms_005_20hz_kpa=float(np.sqrt(energy(freq,psd,.05,20))))
                        for band_low, band_high in BANDS:
                            row[f"rms_{band_low:g}_{band_high:g}hz_kpa"] = float(np.sqrt(energy(freq,psd,band_low,band_high)))
                        row.update({f"ctrl{v}": float(commands[start, v-1]) for v in range(1,7)})
                        rows.append(row)
        for valve in range(1, 7):
            report_path = Path(f"../pneu_tune/exp/lib3_all_valves_feasible_20261022/{kind}/v{valve}/report.json")
            pars = json.loads(report_path.read_text())["parameters"]
            for direction, wn, zeta in (("up", *pars[-4:-2]), ("down", *pars[-2:])):
                parameter_rows.append(dict(type=kind, valve=valve, direction=direction,
                                           wn_rad_per_sec=wn, zeta=zeta,
                                           isolated_damped_frequency_hz=wn*np.sqrt(max(0., 1-zeta*zeta))/(2*np.pi),
                                           isolated_underdamped=bool(zeta < 1)))
        datasets[kind] = (raw, replay)
        lo, hi = (99.0402, 129.0401) if kind == "type1" else (186.0402, 196.0401)
        subset = replay[(replay.source_time >= lo) & (replay.source_time <= hi)]
        spectrum_examples[kind] = (lo, hi, subset)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / f"oscillation_{args.variant}_holds.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / f"oscillation_{args.variant}_valve_dynamics.csv", index=False)

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="col")
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    examples = {}
    for col, kind in enumerate(("type1", "type2")):
        low, high, data = spectrum_examples[kind]
        raw, replay = datasets[kind]
        commands = raw[(raw.time >= low) & (raw.time <= high)]
        for model, ls in (("real", "-"), ("sim", "--")):
            t = data.source_time.to_numpy()
            axes[0,col].plot(t, data[f"{model}_act_pos_press"], ls, label=f"{model} actuator +", lw=1)
            dp3 = data[f"{model}_press_pos"]-data[f"{model}_act_pos_press"]
            dp4 = data[f"{model}_act_pos_press"]-101.325
            axes[1,col].plot(t, dp3, ls, label=f"{model} dP3", lw=1)
            axes[1,col].plot(t, dp4, ls, label=f"{model} dP4", lw=.8)
            axes[2,col].plot(t, data[f"{model}_flow3"], ls, label=f"{model} Q3", lw=1)
            axes[2,col].plot(t, data[f"{model}_flow4"], ls, label=f"{model} Q4", lw=1)
        for valve in range(1,7):
            axes[3,col].plot(commands.time, commands[f"ctrl{valve}"], label=f"u{valve}", lw=.8)
        axes[0,col].set_title(f"{kind}: commanded ramp {low:.2f}–{high:.2f} s")
        axes[0,col].axhline(0, color="r", lw=.5)
        axes[1,col].axhline(0, color="r", lw=.5)
        for row, ylabel in enumerate(("absolute pressure [kPa]", "pressure difference [kPa]", "flow [L/min]", "command")):
            axes[row,col].set_ylabel(ylabel)
            axes[row,col].grid(alpha=.25)
            axes[row,col].legend(fontsize=7, ncol=2)
        axes[3,col].set_xlabel("source time [s]")
        examples[kind] = dict(start_sec=low, end_sec=high,
                             actuator_pos_range={model:[float(data[f'{model}_act_pos_press'].min()),float(data[f'{model}_act_pos_press'].max())] for model in ('real','sim')})
        for c, channel in enumerate(CHANNELS):
            for model in ("real", "sim"):
                f, ps = spectrum(data[f"{model}_{channel}"].to_numpy(), fs)
                axes2[col,c].semilogy(f, ps, label=model, lw=1)
            axes2[col,c].set_xlim(.03, 5.)
            axes2[col,c].set_title(f"{kind} {channel}")
            axes2[col,c].set_xlabel("frequency [Hz]")
            axes2[col,c].grid(alpha=.25)
            axes2[col,c].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"oscillation_{args.variant}_holds.png", dpi=150)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / f"oscillation_{args.variant}_spectra.png", dpi=150)

    summaries = []
    for (kind, scope, channel, model), group in frame.groupby(["type", "scope", "channel", "model"]):
        summaries.append(dict(type=kind, scope=scope, channel=channel, model=model, holds=len(group),
                              median_peak_hz=float(group.peak_hz.median()),
                              median_rms_005_20hz_kpa=float(group.rms_005_20hz_kpa.median()),
                              median_rms_3_20hz_kpa=float(group["rms_3_20hz_kpa"].median()),
                              max_range_kpa=float(group.range_kpa.max())))
    audit = dict(variant=args.variant, sampling_hz=fs,
                 command_hold_definition="all six echo controls have total range <=1e-6 throughout the hold; minimum duration 5 s; excludes intentional slow ramps",
                 limitations=["Fixed commands do not imply settled pressure: both 1 s trimmed and tail windows are reported.",
                              "Linear detrending removes a ramp, not nonlinear pneumatic transients.",
                              "No attribution of >25 Hz content can be made from 50 Hz records.",
                              "Periodograms describe oscillations, not by themselves their causal mechanism.",
                              "All results are open-loop command replay; no RL policy is running."],
                 pump_code=dict(rpm=3000, mechanical_hz=50., opposed_piston_events_hz=100.,
                                exact_sampling_alias_hz=0., note="Code sets 3000 RPM; this does not establish actual hardware RPM."),
                 examples=examples, summaries=summaries, hashes=hashes)
    (args.output_dir / f"oscillation_{args.variant}_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(dict(examples=examples, summaries=[x for x in summaries if x['scope']=='tail' and x['channel']=='act_pos_press']),indent=2))


if __name__ == "__main__":
    main()
