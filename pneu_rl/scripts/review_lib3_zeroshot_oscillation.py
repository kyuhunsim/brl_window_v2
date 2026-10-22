"""Read-only review of archived 2026-10-16 EXP and simulator replay CSVs.

Run from repository root. Does not load a policy or connect to hardware.
Pre-fix EXP time is nominal, so no physical-frequency claim is made for it.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


def main():
    out = Path("docs/img/lib3_zeroshot_review_20261016")
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    sources = {}
    for label, stamp in [("new_before", "15_56_56"), ("old_before", "15_57_57"),
                         ("new_after", "16_15_14")]:
        matches = list(Path("pneu_rl/exp").glob(f"*/*261016_{stamp}*/*.csv"))
        if len(matches) != 1:
            raise ValueError((label, matches))
        p = matches[0]
        runs[label] = pd.read_csv(p)
        sources[label] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    rows = []
    for label, d in runs.items():
        t = d.curr_time.to_numpy()
        for lo, hi in [(0, 41), (6, 38), (20, 38)]:
            m = (t >= lo) & (t < hi)
            for side in ["pos", "neg"]:
                e = (d[f"act_{side}_press"] - d[f"ref_act_{side}"])[m].to_numpy()
                rows.append(dict(run=label, start=lo, end=hi, channel=side, n=len(e),
                                 rmse=np.sqrt(np.mean(e**2)), bias=np.mean(e),
                                 error_std=np.std(e)))
    pd.DataFrame(rows).to_csv(out / "tracking.csv", index=False)

    real = runs["new_after"]
    t = real.curr_time.to_numpy()
    grid = np.arange(6, 38, .02)
    spectra = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for name, d in [("Real", real), ("Same-command Sim", pd.read_csv(
            "docs/img/lib3_command_replay_20261016/measured_pressure_reset.csv"))]:
        tt = d.curr_time if name == "Real" else d.time
        yy = d.act_pos_press if name == "Real" else d.sim_act_pos_press
        y = np.interp(grid, tt, yy)
        f, p = welch(y, fs=50, nperseg=512)
        band = (f >= 8) & (f <= 12)
        spectra.append(dict(signal=name, peak_2_20_hz=float(f[(f >= 2) & (f <= 20)][np.argmax(p[(f >= 2) & (f <= 20)])]),
                            rms_8_12_hz=float(np.sqrt(np.sum(p[band]) * (f[1]-f[0])))))
        axes[0].plot(tt, yy, label=name, alpha=.85)
        axes[1].semilogy(f, p, label=name)
    axes[0].plot(t, real.ref_act_pos, label="Reference", color="black")
    axes[0].set(xlim=(20, 22), ylabel="Actuator positive pressure (kPa)")
    axes[1].set(xlim=(2, 20), xlabel="Frequency (Hz)", ylabel="PSD (kPa²/Hz)")
    for col in ["ctrl_ch_pos", "ctrl_act_pos_in", "ctrl_act_pos_out"]:
        axes[2].plot(t, real[col], label=col)
        y = np.interp(grid, t, real[col])
        f, p = welch(y, fs=50, nperseg=512)
        mask = (f >= 2) & (f <= 20)
        spectra.append(dict(signal=col, peak_2_20_hz=float(f[mask][np.argmax(p[mask])]),
                            rms_8_12_hz=float(np.sqrt(np.sum(p[(f >= 8) & (f <= 12)])*(f[1]-f[0])))))
    axes[2].set(xlim=(20, 22), xlabel="Recorded elapsed time (s)", ylabel="Physical command")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=.25)
    fig.savefig(out / "oscillation_vs_replay.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(spectra).to_csv(out / "spectra.csv", index=False)

    replay_rows = []
    channels = ["press_pos", "press_neg", "act_pos_press", "act_neg_press"]
    for mode in ["warm_sim", "measured_pressure_reset"]:
        d = pd.read_csv(f"docs/img/lib3_command_replay_20261016/{mode}.csv")
        m = (t >= 20) & (t < 38)
        for c in channels:
            s = np.interp(t[m], d.time, d[f"sim_{c}"])
            r = real[c].to_numpy()[m]
            replay_rows.append(dict(mode=mode, channel=c, real_mean=r.mean(), sim_mean=s.mean(),
                                    real_minus_sim=(r-s).mean(), rmse=np.sqrt(np.mean((r-s)**2))))
    pd.DataFrame(replay_rows).to_csv(out / "replay_means.csv", index=False)
    dt = np.diff(t)
    summary = {"sources": sources, "new_after_mean_hz": float(1/dt.mean()),
               "new_after_max_interval_ms": float(dt.max()*1000),
               "spectrum_method": "6<=t<38, interpolate to 50Hz, Welch nperseg=512; band RMS sqrt(sum(PSD)*df)",
               "limitations": ["Before-fix EXP timestamps are nominal, not physical time.",
                               "No physical sensor/application timestamps; spectrum may include aliasing.",
                               "Band RMS measures oscillatory energy, not total tracking error.",
                               "Replay CSVs reused, simulator not re-executed by this script."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(spectra).to_string(index=False))
    print(pd.DataFrame(replay_rows).to_string(index=False))


if __name__ == "__main__":
    main()
