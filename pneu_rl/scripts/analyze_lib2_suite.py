import argparse
import os
from datetime import datetime
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from analyze_lib2_baseline import (
    analyze_df,
    compare_pid_rl,
    diagnose_single,
    run_pid,
    run_rl,
)
from pneu_utils.utils import get_pkg_path


REF_MODES = ["random", "cos", "sin", "step"]


def save_ref_plot(
    pid_sim2_df: pd.DataFrame,
    pid_sim_df: pd.DataFrame,
    rl_df: Optional[pd.DataFrame],
    *,
    ref_mode: str,
    out_dir: str,
    pid_sim2_report: dict,
    pid_sim_report: dict,
    rl_report: Optional[dict],
) -> None:
    sim2_pos_color = "red"
    sim2_neg_color = "blue"
    sim_pos_color = "darkorange"
    sim_neg_color = "deepskyblue"
    rl_pos_color = "green"
    rl_neg_color = "purple"

    num_plots = 8 if rl_df is not None else 5
    fig_height = 16 if rl_df is not None else 10
    fig = plt.figure(figsize=(12, fig_height))
    gs = gridspec.GridSpec(num_plots, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)

    ax1.plot(pid_sim2_df["curr_time"], pid_sim2_df["ref_pos"], color="black", label="ref_pos")
    ax1.plot(pid_sim2_df["curr_time"], pid_sim2_df["sen_pos"], color=sim2_pos_color, label="pid_sim2_pos")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax1.set_ylabel("PID Sim2 Pos [kPa]")

    ax2.plot(pid_sim2_df["curr_time"], pid_sim2_df["ref_neg"], color="black", label="ref_neg")
    ax2.plot(pid_sim2_df["curr_time"], pid_sim2_df["sen_neg"], color=sim2_neg_color, label="pid_sim2_neg")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    ax2.set_ylabel("PID Sim2 Neg [kPa]")

    ax3.plot(pid_sim_df["curr_time"], pid_sim_df["ref_pos"], color="black", label="ref_pos")
    ax3.plot(pid_sim_df["curr_time"], pid_sim_df["sen_pos"], color=sim_pos_color, label="pid_sim_pos")
    ax3.grid(True)
    ax3.legend(loc="upper right")
    ax3.set_ylabel("PID Sim Pos [kPa]")

    ax4.plot(pid_sim_df["curr_time"], pid_sim_df["ref_neg"], color="black", label="ref_neg")
    ax4.plot(pid_sim_df["curr_time"], pid_sim_df["sen_neg"], color=sim_neg_color, label="pid_sim_neg")
    ax4.grid(True)
    ax4.legend(loc="upper right")
    ax4.set_ylabel("PID Sim Neg [kPa]")

    if rl_df is not None:
        ax6 = fig.add_subplot(gs[5, 0], sharex=ax1)
        ax7 = fig.add_subplot(gs[6, 0], sharex=ax1)
        ax8 = fig.add_subplot(gs[7, 0], sharex=ax1)

        ax5.plot(rl_df["curr_time"], rl_df["ref_pos"], color="black", label="ref_pos")
        ax5.plot(rl_df["curr_time"], rl_df["sen_pos"], color=rl_pos_color, label="rl_sim2_pos")
        ax6.plot(rl_df["curr_time"], rl_df["ref_neg"], color="black", label="ref_neg")
        ax6.plot(rl_df["curr_time"], rl_df["sen_neg"], color=rl_neg_color, label="rl_sim2_neg")
        ax5.grid(True)
        ax5.legend(loc="upper right")
        ax5.set_ylabel("RL Sim2 Pos [kPa]")

        ax6.grid(True)
        ax6.legend(loc="upper right")
        ax6.set_ylabel("RL Sim2 Neg [kPa]")

        ax7.plot(pid_sim2_df["curr_time"], pid_sim2_df["ctrl_pos"], color=sim2_pos_color, label="pid_sim2_ctrl_pos")
        ax7.plot(pid_sim2_df["curr_time"], pid_sim2_df["ctrl_neg"], color=sim2_neg_color, label="pid_sim2_ctrl_neg")
        ax7.plot(pid_sim_df["curr_time"], pid_sim_df["ctrl_pos"], color=sim_pos_color, linestyle="--", label="pid_sim_ctrl_pos")
        ax7.plot(pid_sim_df["curr_time"], pid_sim_df["ctrl_neg"], color=sim_neg_color, linestyle="--", label="pid_sim_ctrl_neg")
        ax7.grid(True)
        ax7.legend(loc="upper right")
        ax7.set_ylabel("PID Control")

        ax8.plot(rl_df["curr_time"], rl_df["ctrl_pos"], color=rl_pos_color, label="rl_sim2_ctrl_pos")
        ax8.plot(rl_df["curr_time"], rl_df["ctrl_neg"], color=rl_neg_color, label="rl_sim2_ctrl_neg")
        ax8.grid(True)
        ax8.legend(loc="upper right")
        ax8.set_ylabel("RL Control")
        ax8.set_xlabel("Time [sec]")
    else:
        ax5.plot(pid_sim2_df["curr_time"], pid_sim2_df["ctrl_pos"], color=sim2_pos_color, label="pid_sim2_ctrl_pos")
        ax5.plot(pid_sim2_df["curr_time"], pid_sim2_df["ctrl_neg"], color=sim2_neg_color, label="pid_sim2_ctrl_neg")
        ax5.plot(pid_sim_df["curr_time"], pid_sim_df["ctrl_pos"], color=sim_pos_color, linestyle="--", label="pid_sim_ctrl_pos")
        ax5.plot(pid_sim_df["curr_time"], pid_sim_df["ctrl_neg"], color=sim_neg_color, linestyle="--", label="pid_sim_ctrl_neg")
        ax5.grid(True)
        ax5.legend(loc="upper right")
        ax5.set_ylabel("PID Control")
        ax5.set_xlabel("Time [sec]")

    sim2_metrics = pid_sim2_report["metrics"]
    sim_metrics = pid_sim_report["metrics"]
    title = (
        f"{ref_mode} | PID sim2 pos={sim2_metrics['rmse_pos']:.3f}, neg={sim2_metrics['rmse_neg']:.3f}"
        f" | PID sim pos={sim_metrics['rmse_pos']:.3f}, neg={sim_metrics['rmse_neg']:.3f}"
    )
    if rl_report is not None:
        rl_metrics = rl_report["metrics"]
        title += f" | RL sim2 pos={rl_metrics['rmse_pos']:.3f}, neg={rl_metrics['rmse_neg']:.3f}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{ref_mode}_diagnosis.png", dpi=150)
    plt.close(fig)


def flatten_run(prefix: str, report: dict) -> dict:
    metrics = report["metrics"]
    return dict(
        **{f"{prefix}_{k}": v for k, v in metrics.items()},
        **{
            f"{prefix}_gain_pos": report["gain"]["pos"],
            f"{prefix}_gain_neg": report["gain"]["neg"],
            f"{prefix}_lag_pos_sec": report["lag"]["pos"]["lag_sec"],
            f"{prefix}_lag_neg_sec": report["lag"]["neg"]["lag_sec"],
            f"{prefix}_lag_pos_corr": report["lag"]["pos"]["corr"],
            f"{prefix}_lag_neg_corr": report["lag"]["neg"]["corr"],
            f"{prefix}_ctrl_pos_low_pct": report["saturation"]["ctrl_pos"]["low_pct"],
            f"{prefix}_ctrl_pos_high_pct": report["saturation"]["ctrl_pos"]["high_pct"],
            f"{prefix}_ctrl_neg_low_pct": report["saturation"]["ctrl_neg"]["low_pct"],
            f"{prefix}_ctrl_neg_high_pct": report["saturation"]["ctrl_neg"]["high_pct"],
            f"{prefix}_dpdt_pos_rise_p95": report["dpdt"]["pos"]["rise_p95"],
            f"{prefix}_dpdt_pos_fall_p05": report["dpdt"]["pos"]["fall_p05"],
            f"{prefix}_dpdt_neg_rise_p95": report["dpdt"]["neg"]["rise_p95"],
            f"{prefix}_dpdt_neg_fall_p05": report["dpdt"]["neg"]["fall_p05"],
            f"{prefix}_wrong_pos_pct": report["alignment"]["pos"]["wrong_pct"],
            f"{prefix}_wrong_neg_pct": report["alignment"]["neg"]["wrong_pct"],
        },
    )


def analyze_one_ref(args: argparse.Namespace, ref_mode: str, out_dir: str) -> dict:
    run_args = argparse.Namespace(**vars(args))
    run_args.ref = ref_mode

    print(f"\n[RUN] {ref_mode}")
    pid_sim2_df = run_pid(run_args, sim_kind="sim2")
    pid_sim_df = run_pid(run_args, sim_kind="sim")
    rl_df = run_rl(run_args) if args.plot_mode == "pid_rl" else None

    pid_sim2_report = analyze_df(
        pid_sim2_df,
        skip_sec=args.skip_sec,
        low_thr=args.low_thr,
        high_thr=args.high_thr,
        err_thr=args.err_thr,
        ctrl_mid=args.ctrl_mid,
        max_lag_sec=args.max_lag_sec,
    )

    pid_sim_report = analyze_df(
        pid_sim_df,
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

    diagnosis = diagnose_single("PID sim2", pid_sim2_report)
    diagnosis.extend(diagnose_single("PID sim", pid_sim_report))
    if rl_report is not None:
        diagnosis.extend(diagnose_single("RL sim2", rl_report))
        diagnosis.extend(compare_pid_rl(pid_sim2_report, rl_report))

    pid_sim2_df.to_csv(f"{out_dir}/{ref_mode}_pid_sim2.csv", index=False)
    pid_sim_df.to_csv(f"{out_dir}/{ref_mode}_pid_sim.csv", index=False)
    if rl_df is not None:
        rl_df.to_csv(f"{out_dir}/{ref_mode}_rl_sim2.csv", index=False)

    save_ref_plot(
        pid_sim2_df,
        pid_sim_df,
        rl_df,
        ref_mode=ref_mode,
        out_dir=out_dir,
        pid_sim2_report=pid_sim2_report,
        pid_sim_report=pid_sim_report,
        rl_report=rl_report,
    )

    row = dict(ref=ref_mode)
    row.update(flatten_run("pid_sim2", pid_sim2_report))
    row.update(flatten_run("pid_sim", pid_sim_report))
    if rl_report is not None:
        row.update(flatten_run("rl_sim2", rl_report))
    row["diagnosis"] = " | ".join(diagnosis)

    return dict(
        row=row,
        report=dict(
            pid_sim2=pid_sim2_report,
            pid_sim=pid_sim_report,
            rl_sim2=rl_report,
            diagnosis=diagnosis,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", choices=REF_MODES, default=REF_MODES)
    ap.add_argument("--plot-mode", choices=["pid_rl", "pid"], default="pid_rl")
    ap.add_argument("--model-name", default="0427_lib2_Ours_2")
    ap.add_argument("--max-time", type=float, default=100.0)
    ap.add_argument("--freq", type=float, default=50.0)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--scale", action="store_true", default=True)
    ap.add_argument("--no-scale", dest="scale", action="store_false")
    ap.add_argument("--low-thr", type=float, default=0.7)
    ap.add_argument("--high-thr", type=float, default=1.0)
    ap.add_argument("--ctrl-mid", type=float, default=0.85)
    ap.add_argument("--err-thr", type=float, default=5.0)
    ap.add_argument("--max-lag-sec", type=float, default=10.0)
    ap.add_argument("--skip-sec", type=float, default=2.0)
    ap.add_argument("--save-name", default=None)
    args = ap.parse_args()

    if args.save_name:
        save_name = args.save_name
    else:
        stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        save_name = f"{stamp}_lib2_{args.plot_mode}_diagnosis_suite"

    out_dir = f'{get_pkg_path("pneu_rl")}/exp/{save_name}'
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    reports = {}
    for ref_mode in args.refs:
        result = analyze_one_ref(args, ref_mode, out_dir)
        rows.append(result["row"])
        reports[ref_mode] = result["report"]

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/summary.csv", index=False)

    suite = dict(
        args=vars(args),
        reports=reports,
    )
    with open(f"{out_dir}/summary.yaml", "w") as f:
        yaml.dump(suite, f)

    print(f"\n[INFO] Saved suite diagnosis: {out_dir}")


if __name__ == "__main__":
    main()
