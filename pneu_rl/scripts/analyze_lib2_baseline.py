import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from baseline_lib2 import (
    PID_GAINS,
    REF_KWARGS,
    calc_metrics,
    run_pid,
    run_rl,
)
from pneu_utils.utils import get_pkg_path


def robust_amplitude(x: np.ndarray) -> float:
    return float((np.percentile(x, 95) - np.percentile(x, 5)) / 2.0)


def calc_gain(ref: np.ndarray, sen: np.ndarray) -> float:
    ref_amp = robust_amplitude(ref)
    sen_amp = robust_amplitude(sen)
    if ref_amp < 1e-9:
        return 0.0
    return float(sen_amp / ref_amp)


def calc_lag(
    time: np.ndarray,
    ref: np.ndarray,
    sen: np.ndarray,
    max_lag_sec: float,
) -> dict:
    dt = float(np.median(np.diff(time)))
    max_lag = max(1, int(max_lag_sec / max(dt, 1e-9)))

    ref0 = ref - np.mean(ref)
    sen0 = sen - np.mean(sen)
    best_lag = 0
    best_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a = ref0[:-lag]
            b = sen0[lag:]
        elif lag < 0:
            a = ref0[-lag:]
            b = sen0[:lag]
        else:
            a = ref0
            b = sen0

        if len(a) < 3:
            continue

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            continue

        corr = float(np.dot(a, b) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return dict(
        lag_sec=float(best_lag * dt),
        corr=float(best_corr),
    )


def calc_saturation(ctrl: np.ndarray, low_thr: float, high_thr: float) -> dict:
    return dict(
        low_pct=float(np.mean(ctrl <= low_thr) * 100.0),
        high_pct=float(np.mean(ctrl >= high_thr) * 100.0),
        min=float(np.min(ctrl)),
        max=float(np.max(ctrl)),
        mean=float(np.mean(ctrl)),
    )


def calc_dpdt(time: np.ndarray, sen: np.ndarray) -> dict:
    dt = np.diff(time)
    dp = np.diff(sen)
    valid = dt > 1e-9
    if not np.any(valid):
        return dict(rise_p95=0.0, fall_p05=0.0, max=0.0, min=0.0)

    rate = dp[valid] / dt[valid]
    return dict(
        rise_p95=float(np.percentile(rate, 95)),
        fall_p05=float(np.percentile(rate, 5)),
        max=float(np.max(rate)),
        min=float(np.min(rate)),
    )


def calc_alignment(
    ref: np.ndarray,
    sen: np.ndarray,
    ctrl: np.ndarray,
    *,
    axis: str,
    err_thr: float,
    ctrl_mid: float,
) -> dict:
    err = ref - sen
    mask = np.abs(err) >= err_thr
    if not np.any(mask):
        return dict(
            active_pct=0.0,
            wrong_pct=0.0,
            timid_pct=0.0,
        )

    active_err = err[mask]
    active_ctrl = ctrl[mask]

    if axis == "pos":
        wrong = ((active_err > 0) & (active_ctrl > ctrl_mid)) | (
            (active_err < 0) & (active_ctrl < ctrl_mid)
        )
    elif axis == "neg":
        wrong = ((active_err > 0) & (active_ctrl < ctrl_mid)) | (
            (active_err < 0) & (active_ctrl > ctrl_mid)
        )
    else:
        raise ValueError(f"Unknown axis: {axis}")

    timid = np.abs(active_ctrl - ctrl_mid) < 0.05

    return dict(
        active_pct=float(np.mean(mask) * 100.0),
        wrong_pct=float(np.mean(wrong) * 100.0),
        timid_pct=float(np.mean(timid) * 100.0),
    )


def range_stats(x: np.ndarray) -> dict:
    return dict(
        min=float(np.min(x)),
        max=float(np.max(x)),
        p05=float(np.percentile(x, 5)),
        p95=float(np.percentile(x, 95)),
    )


def analyze_df(
    df: pd.DataFrame,
    *,
    skip_sec: float,
    low_thr: float,
    high_thr: float,
    err_thr: float,
    ctrl_mid: float,
    max_lag_sec: float,
) -> dict:
    if skip_sec > 0.0:
        start_time = float(df["curr_time"].iloc[0])
        filtered = df[df["curr_time"] >= start_time + skip_sec]
        if len(filtered) >= 10:
            df = filtered

    time = df["curr_time"].to_numpy(dtype=np.float64)
    ref_pos = df["ref_pos"].to_numpy(dtype=np.float64)
    ref_neg = df["ref_neg"].to_numpy(dtype=np.float64)
    sen_pos = df["sen_pos"].to_numpy(dtype=np.float64)
    sen_neg = df["sen_neg"].to_numpy(dtype=np.float64)
    ctrl_pos = df["ctrl_pos"].to_numpy(dtype=np.float64)
    ctrl_neg = df["ctrl_neg"].to_numpy(dtype=np.float64)

    return dict(
        metrics=calc_metrics(df),
        range=dict(
            ref_pos=range_stats(ref_pos),
            ref_neg=range_stats(ref_neg),
            sen_pos=range_stats(sen_pos),
            sen_neg=range_stats(sen_neg),
        ),
        gain=dict(
            pos=calc_gain(ref_pos, sen_pos),
            neg=calc_gain(ref_neg, sen_neg),
        ),
        lag=dict(
            pos=calc_lag(time, ref_pos, sen_pos, max_lag_sec),
            neg=calc_lag(time, ref_neg, sen_neg, max_lag_sec),
        ),
        saturation=dict(
            ctrl_pos=calc_saturation(ctrl_pos, low_thr, high_thr),
            ctrl_neg=calc_saturation(ctrl_neg, low_thr, high_thr),
        ),
        dpdt=dict(
            pos=calc_dpdt(time, sen_pos),
            neg=calc_dpdt(time, sen_neg),
        ),
        alignment=dict(
            pos=calc_alignment(
                ref_pos,
                sen_pos,
                ctrl_pos,
                axis="pos",
                err_thr=err_thr,
                ctrl_mid=ctrl_mid,
            ),
            neg=calc_alignment(
                ref_neg,
                sen_neg,
                ctrl_neg,
                axis="neg",
                err_thr=err_thr,
                ctrl_mid=ctrl_mid,
            ),
        ),
    )


def diagnose_single(name: str, report: dict) -> list[str]:
    lines = []
    metrics = report["metrics"]
    gain = report["gain"]
    lag = report["lag"]
    sat = report["saturation"]
    align = report["alignment"]

    if gain["pos"] < 0.6:
        lines.append(f"{name}: pos gain is low ({gain['pos']:.2f}); pos pressure amplitude is not following ref well.")
    if gain["neg"] < 0.6:
        lines.append(f"{name}: neg gain is low ({gain['neg']:.2f}); neg pressure range/authority may be limited.")

    if lag["pos"]["lag_sec"] > 1.0:
        lines.append(f"{name}: pos lag is large ({lag['pos']['lag_sec']:.2f}s); bandwidth/delay is visible.")
    if lag["neg"]["lag_sec"] > 1.0:
        lines.append(f"{name}: neg lag is large ({lag['neg']['lag_sec']:.2f}s); bandwidth/delay is visible.")

    if sat["ctrl_pos"]["high_pct"] > 40.0 or sat["ctrl_pos"]["low_pct"] > 40.0:
        lines.append(
            f"{name}: ctrl_pos is saturated often "
            f"(low {sat['ctrl_pos']['low_pct']:.1f}%, high {sat['ctrl_pos']['high_pct']:.1f}%)."
        )
    if sat["ctrl_neg"]["high_pct"] > 40.0 or sat["ctrl_neg"]["low_pct"] > 40.0:
        lines.append(
            f"{name}: ctrl_neg is saturated often "
            f"(low {sat['ctrl_neg']['low_pct']:.1f}%, high {sat['ctrl_neg']['high_pct']:.1f}%)."
        )

    if align["pos"]["wrong_pct"] > 25.0:
        lines.append(f"{name}: pos control direction/timing looks suspicious (wrong {align['pos']['wrong_pct']:.1f}%).")
    if align["neg"]["wrong_pct"] > 25.0:
        lines.append(f"{name}: neg control direction/timing looks suspicious (wrong {align['neg']['wrong_pct']:.1f}%).")

    if metrics["rmse_pos"] > 20.0 and gain["pos"] >= 0.7 and lag["pos"]["lag_sec"] > 1.0:
        lines.append(f"{name}: pos error is likely dominated by phase lag rather than pure amplitude loss.")
    if metrics["rmse_neg"] > 8.0 and gain["neg"] < 0.7:
        lines.append(f"{name}: neg error is likely dominated by limited amplitude/range tracking.")

    return lines


def compare_pid_rl(pid_report: dict, rl_report: dict) -> list[str]:
    lines = []
    pid = pid_report["metrics"]
    rl = rl_report["metrics"]

    if pid["rmse_pos"] < 0.75 * rl["rmse_pos"]:
        lines.append("PID tracks pos clearly better than RL; policy/observation/reward is likely limiting pos.")
    elif rl["rmse_pos"] < 0.75 * pid["rmse_pos"]:
        lines.append("RL tracks pos clearly better than PID; PID tuning is not an upper bound for pos here.")
    else:
        lines.append("PID and RL pos RMSE are similar; plant/ref difficulty is mixed with controller quality.")

    if pid["rmse_neg"] < 0.75 * rl["rmse_neg"]:
        lines.append("PID tracks neg clearly better than RL; policy/observation/reward is likely limiting neg.")
    elif rl["rmse_neg"] < 0.75 * pid["rmse_neg"]:
        lines.append("RL tracks neg clearly better than PID; PID tuning is not an upper bound for neg here.")
    else:
        lines.append("PID and RL neg RMSE are similar; plant/ref difficulty is mixed with controller quality.")

    return lines


def print_summary(report: dict) -> None:
    for name in ["pid", "rl"]:
        if name not in report["runs"] or report["runs"][name] is None:
            continue

        run = report["runs"][name]
        m = run["metrics"]
        print(f"\n[{name.upper()}]")
        print(f"RMSE pos/neg: {m['rmse_pos']:.3f} / {m['rmse_neg']:.3f}")
        print(f"MAE  pos/neg: {m['mae_pos']:.3f} / {m['mae_neg']:.3f}")
        print(f"Gain pos/neg: {run['gain']['pos']:.3f} / {run['gain']['neg']:.3f}")
        print(
            "Lag  pos/neg: "
            f"{run['lag']['pos']['lag_sec']:.3f}s / {run['lag']['neg']['lag_sec']:.3f}s"
        )
        print(
            "Sat ctrl_pos low/high: "
            f"{run['saturation']['ctrl_pos']['low_pct']:.1f}% / {run['saturation']['ctrl_pos']['high_pct']:.1f}%"
        )
        print(
            "Sat ctrl_neg low/high: "
            f"{run['saturation']['ctrl_neg']['low_pct']:.1f}% / {run['saturation']['ctrl_neg']['high_pct']:.1f}%"
        )
        print(
            "dPdt pos rise/fall p95/p05: "
            f"{run['dpdt']['pos']['rise_p95']:.3f} / {run['dpdt']['pos']['fall_p05']:.3f} kPa/s"
        )
        print(
            "dPdt neg rise/fall p95/p05: "
            f"{run['dpdt']['neg']['rise_p95']:.3f} / {run['dpdt']['neg']['fall_p05']:.3f} kPa/s"
        )
        print(
            "Wrong direction pos/neg: "
            f"{run['alignment']['pos']['wrong_pct']:.1f}% / {run['alignment']['neg']['wrong_pct']:.1f}%"
        )

    print("\n[DIAGNOSIS]")
    for line in report["diagnosis"]:
        print(f"- {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-mode", choices=["pid_rl", "pid"], default="pid_rl")
    ap.add_argument("--model-name", default="0427_lib2_Ours_2")
    ap.add_argument("--ref", choices=["random", "sin", "cos", "step"], default="random")
    ap.add_argument("--max-time", type=float, default=100.0)
    ap.add_argument("--freq", type=float, default=50.0)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--scale", action="store_true", default=True)
    ap.add_argument("--no-scale", dest="scale", action="store_false")
    ap.add_argument("--low-thr", type=float, default=0.72)
    ap.add_argument("--high-thr", type=float, default=0.98)
    ap.add_argument("--ctrl-mid", type=float, default=0.85)
    ap.add_argument("--err-thr", type=float, default=5.0)
    ap.add_argument("--max-lag-sec", type=float, default=10.0)
    ap.add_argument("--skip-sec", type=float, default=2.0)
    ap.add_argument("--save-name", default=None)
    args = ap.parse_args()

    pid_df = run_pid(args)
    rl_df = run_rl(args) if args.plot_mode == "pid_rl" else None

    pid_report = analyze_df(
        pid_df,
        skip_sec=args.skip_sec,
        low_thr=args.low_thr,
        high_thr=args.high_thr,
        err_thr=args.err_thr,
        ctrl_mid=args.ctrl_mid,
        max_lag_sec=args.max_lag_sec,
    )
    rl_report = None
    if rl_df is not None:
        rl_report = analyze_df(
            rl_df,
            skip_sec=args.skip_sec,
            low_thr=args.low_thr,
            high_thr=args.high_thr,
            err_thr=args.err_thr,
            ctrl_mid=args.ctrl_mid,
            max_lag_sec=args.max_lag_sec,
        )

    diagnosis = diagnose_single("PID", pid_report)
    if rl_report is not None:
        diagnosis.extend(diagnose_single("RL", rl_report))
        diagnosis.extend(compare_pid_rl(pid_report, rl_report))

    report = dict(
        args=vars(args),
        ref=dict(mode=args.ref, kwargs=REF_KWARGS[args.ref]),
        pid_gains=PID_GAINS,
        runs=dict(
            pid=pid_report,
            rl=rl_report,
        ),
        diagnosis=diagnosis,
    )

    if args.save_name:
        save_name = args.save_name
    else:
        stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        save_name = f"{stamp}_lib2_{args.ref}_{args.plot_mode}_diagnosis"

    out_dir = f'{get_pkg_path("pneu_rl")}/exp/{save_name}'
    os.makedirs(out_dir, exist_ok=True)
    pid_df.to_csv(f"{out_dir}/{save_name}_pid.csv", index=False)
    if rl_df is not None:
        rl_df.to_csv(f"{out_dir}/{save_name}_rl.csv", index=False)
    with open(f"{out_dir}/diagnosis.yaml", "w") as f:
        yaml.dump(report, f)

    print_summary(report)
    print(f"\n[INFO] Saved diagnosis: {out_dir}")


if __name__ == "__main__":
    main()
