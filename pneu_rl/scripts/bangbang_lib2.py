import argparse
import os
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim import PneuSim as PneuSimBase
from pneu_env.sim2 import PneuSim as PneuSim2
from pneu_utils.utils import get_pkg_path


ATM = 101.325

BANGBANG_CFG = dict(
    sims = ["sim2", "sim"],  # choose any of: "sim2", "sim"
    freq = 50.0,
    delay = 0.1,
    scale = True,
    pos = dict(
        mode = "bangbang",  # "bangbang" or "fixed"
        fixed = 0.85,
        min = 0.85,
        max = 1.0,
        phase = "normal",  # "normal" or "inverse"
    ),
    neg = dict(
        mode = "fixed",  # "bangbang" or "fixed"
        fixed = 1.0,
        min = 0.85,
        max = 1.0,
        phase = "inverse",  # opposite to pos by default
    ),
    periods = None,
    start_period = 10.0,
    end_period = 0.5,
    period_ratio = 0.5,
    cycles_per_period = 2.0,
    valve_debug = True,
    save_name = None,
)

VALVE_DEBUG_COLUMNS = [
    f"{side}_{name}"
    for side in ("pos", "neg")
    for name in (
        "u_eff",
        "current",
        "state_curr",
        "z",
        "force_net",
        "area_eff",
        "q_static_lpm",
        "q_pred_lpm",
        "mdot",
    )
]


def make_simulator(args: argparse.Namespace, sim_name: str):
    if sim_name == "sim2":
        sim_cls = PneuSim2
    elif sim_name == "sim":
        sim_cls = PneuSimBase
    else:
        raise ValueError(f"Unknown sim: {sim_name}")

    sim = sim_cls(
        freq=args.freq,
        delay=args.delay,
        noise=False,
        scale=args.scale,
    )
    sim.set_init_press(ATM, ATM)
    return sim


def command_to_action(ctrl: np.ndarray, *, scale: bool) -> np.ndarray:
    if scale:
        return (ctrl - 0.85) / 0.15
    return 2.0 * ctrl - 1.0


def build_period_schedule(args: argparse.Namespace) -> list[float]:
    if args.periods:
        return [float(p) for p in args.periods]

    periods = []
    period = args.start_period
    while period >= args.end_period:
        periods.append(float(period))
        period *= args.period_ratio
    if periods[-1] != args.end_period:
        periods.append(float(args.end_period))
    return periods


def channel_command(
    elapsed: float,
    period: float,
    cfg: dict,
) -> float:
    if cfg["mode"] == "fixed":
        return float(cfg["fixed"])
    if cfg["mode"] != "bangbang":
        raise ValueError(f"Unknown channel mode: {cfg['mode']}")

    first_half = (elapsed % period) < (0.5 * period)
    if cfg["phase"] == "inverse":
        first_half = not first_half
    elif cfg["phase"] != "normal":
        raise ValueError(f"Unknown phase: {cfg['phase']}")

    return float(cfg["min"] if first_half else cfg["max"])


def bangbang_command(
    elapsed: float,
    period: float,
) -> np.ndarray:
    ctrl_pos = channel_command(elapsed, period, BANGBANG_CFG["pos"])
    ctrl_neg = channel_command(elapsed, period, BANGBANG_CFG["neg"])
    return np.array([ctrl_pos, ctrl_neg], dtype=np.float64)


def run_single(args: argparse.Namespace, sim_name: str) -> tuple[pd.DataFrame, list[float]]:
    sim = make_simulator(args, sim_name)
    periods = build_period_schedule(args)

    rows = []
    curr_time = 0.0
    prev_time = None
    prev_pos = None
    prev_neg = None
    segment_start = 0.0
    dt = 1.0 / float(args.freq)
    goal = np.array([ATM, ATM], dtype=np.float64)

    for period in periods:
        segment_duration = period * args.cycles_per_period
        while curr_time < segment_start + segment_duration:
            elapsed = curr_time - segment_start
            ctrl = bangbang_command(elapsed, period)
            action = command_to_action(ctrl, scale=args.scale)
            obs, info = sim.observe(action, goal)
            o = info["Observation"]
            curr_time = float(obs[0])
            sen_pos = float(o["sen_pos"])
            sen_neg = float(o["sen_neg"])
            flow = sim.get_mean_mass_flowrate()
            if args.valve_debug and hasattr(sim, "get_valve_debug"):
                valve_debug = sim.get_valve_debug()
            else:
                valve_debug = {name: np.nan for name in VALVE_DEBUG_COLUMNS}

            if prev_time is None or curr_time <= prev_time:
                dposdt = 0.0
                dnegdt = 0.0
            else:
                dt_obs = curr_time - prev_time
                dposdt = (sen_pos - prev_pos) / dt_obs
                dnegdt = (sen_neg - prev_neg) / dt_obs

            prev_time = curr_time
            prev_pos = sen_pos
            prev_neg = sen_neg

            row = dict(
                sim_name=sim_name,
                curr_time=o["curr_time"],
                period=period,
                sen_pos=sen_pos,
                sen_neg=sen_neg,
                dPpos_dt=dposdt,
                dPneg_dt=dnegdt,
                mean_pump_in=flow["pump_in"],
                mean_pump_out=flow["pump_out"],
                mean_valve_pos=flow["valve_pos"],
                mean_valve_neg=flow["valve_neg"],
                action_pos=float(action[0]),
                action_neg=float(action[1]),
                ctrl_pos=o["ctrl_pos"],
                ctrl_neg=o["ctrl_neg"],
            )
            row.update(valve_debug)
            rows.append(row)

            if dt <= 0.0:
                raise ValueError("freq must be positive")

        segment_start = curr_time

    return pd.DataFrame(rows), periods


def run(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], list[float]]:
    dfs = {}
    periods = None
    for sim_name in args.sims:
        df, sim_periods = run_single(args, sim_name)
        dfs[sim_name] = df
        if periods is None:
            periods = sim_periods
    return dfs, periods


def calc_summary(df: pd.DataFrame) -> dict:
    return dict(
        pos_min=float(df["sen_pos"].min()),
        pos_max=float(df["sen_pos"].max()),
        neg_min=float(df["sen_neg"].min()),
        neg_max=float(df["sen_neg"].max()),
        dPpos_dt_min=float(df["dPpos_dt"].min()),
        dPpos_dt_max=float(df["dPpos_dt"].max()),
        dPneg_dt_min=float(df["dPneg_dt"].min()),
        dPneg_dt_max=float(df["dPneg_dt"].max()),
        mean_pump_in_min=float(df["mean_pump_in"].min()),
        mean_pump_in_max=float(df["mean_pump_in"].max()),
        mean_pump_out_min=float(df["mean_pump_out"].min()),
        mean_pump_out_max=float(df["mean_pump_out"].max()),
        mean_valve_pos_min=float(df["mean_valve_pos"].min()),
        mean_valve_pos_max=float(df["mean_valve_pos"].max()),
        mean_valve_neg_min=float(df["mean_valve_neg"].min()),
        mean_valve_neg_max=float(df["mean_valve_neg"].max()),
        ctrl_pos_min=float(df["ctrl_pos"].min()),
        ctrl_pos_max=float(df["ctrl_pos"].max()),
        ctrl_neg_min=float(df["ctrl_neg"].min()),
        ctrl_neg_max=float(df["ctrl_neg"].max()),
    )


def has_finite_valve_debug(df: pd.DataFrame) -> bool:
    if "pos_q_static_lpm" not in df.columns or "neg_q_static_lpm" not in df.columns:
        return False
    vals = df[["pos_q_static_lpm", "pos_q_pred_lpm", "neg_q_static_lpm", "neg_q_pred_lpm"]].to_numpy()
    return bool(np.isfinite(vals).any())


def add_period_lines(axes: list, df: pd.DataFrame, periods: list[float]) -> None:
    for period in periods[1:]:
        idx = df.index[df["period"] == period]
        if len(idx) == 0:
            continue
        t0 = float(df.loc[idx[0], "curr_time"])
        for ax in axes:
            ax.axvline(t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)


def save_valve_debug_plot(
    df: pd.DataFrame,
    *,
    sim_name: str,
    out_dir: str,
    save_name: str,
    periods: list[float],
) -> None:
    if not has_finite_valve_debug(df):
        return

    fig, axes = plt.subplots(8, 1, figsize=(14, 18), sharex=True)
    pos_color = "red"
    neg_color = "blue"
    time = df["curr_time"]

    axes[0].plot(time, df["ctrl_pos"], color=pos_color, label="ctrl_pos")
    axes[0].plot(time, df["ctrl_neg"], color=neg_color, label="ctrl_neg")
    axes[0].plot(time, df["pos_u_eff"], color=pos_color, linestyle="--", label="pos_u_eff")
    axes[0].plot(time, df["neg_u_eff"], color=neg_color, linestyle="--", label="neg_u_eff")
    axes[0].set_ylabel("Command")

    axes[1].plot(time, df["pos_current"], color=pos_color, label="pos_current")
    axes[1].plot(time, df["neg_current"], color=neg_color, label="neg_current")
    axes[1].set_ylabel("Current [A]")

    axes[2].plot(time, df["pos_force_net"], color=pos_color, label="pos_force_net")
    axes[2].plot(time, df["neg_force_net"], color=neg_color, label="neg_force_net")
    axes[2].set_ylabel("Force Net")

    axes[3].plot(time, df["pos_z"], color=pos_color, label="pos_z")
    axes[3].plot(time, df["neg_z"], color=neg_color, label="neg_z")
    axes[3].set_ylabel("Hysteresis z")

    axes[4].plot(time, df["pos_area_eff"], color=pos_color, label="pos_area_eff")
    axes[4].plot(time, df["neg_area_eff"], color=neg_color, label="neg_area_eff")
    axes[4].set_ylabel("Area Eff")

    axes[5].plot(time, df["pos_q_static_lpm"], color=pos_color, linestyle="--", label="pos_q_static_lpm")
    axes[5].plot(time, df["pos_q_pred_lpm"], color=pos_color, label="pos_q_pred_lpm")
    axes[5].plot(time, df["neg_q_static_lpm"], color=neg_color, linestyle="--", label="neg_q_static_lpm")
    axes[5].plot(time, df["neg_q_pred_lpm"], color=neg_color, label="neg_q_pred_lpm")
    axes[5].set_ylabel("Valve q [LPM]")

    axes[6].plot(time, df["pos_mdot"], color=pos_color, label="pos_mdot")
    axes[6].plot(time, df["neg_mdot"], color=neg_color, label="neg_mdot")
    axes[6].set_ylabel("mdot")

    axes[7].plot(time, df["mean_valve_pos"], color=pos_color, label="mean_valve_pos")
    axes[7].plot(time, df["mean_valve_neg"], color=neg_color, label="mean_valve_neg")
    axes[7].plot(time, df["dPpos_dt"], color=pos_color, linestyle="--", label="dPpos_dt")
    axes[7].plot(time, df["dPneg_dt"], color=neg_color, linestyle="--", label="dPneg_dt")
    axes[7].set_ylabel("Mean Flow / dPdt")
    axes[7].set_xlabel("Time [sec]")

    for ax in axes:
        ax.grid(True)
        ax.legend(loc="upper right", ncol=2)

    add_period_lines(list(axes), df, periods)
    fig.suptitle(f"{save_name} | {sim_name} valve debug")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{save_name}_{sim_name}_valve_debug.png", dpi=150)
    plt.close(fig)


def is_channel_active(channel: str) -> bool:
    return BANGBANG_CFG[channel]["mode"] != "fixed"


def legend_above(ax, *, ncol: int = 1) -> None:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        ncol=ncol,
        fontsize=8,
    )


def save_overview_plot(
    df: pd.DataFrame,
    *,
    sim_name: str,
    out_dir: str,
    save_name: str,
    periods: list[float],
    args: argparse.Namespace,
) -> None:
    pos_active = is_channel_active("pos")
    neg_active = is_channel_active("neg")
    pos_color, neg_color = {
        "sim2": ("red", "blue"),
        "sim": ("darkorange", "deepskyblue"),
    }.get(sim_name, ("red", "blue"))

    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    time = df["curr_time"]
    show_valve_debug = args.valve_debug and has_finite_valve_debug(df)

    axes[0].plot(time, df["sen_pos"], color=pos_color, label=f"{sim_name}_pos")
    axes[0].set_ylabel(f"{sim_name} Pos [kPa]")

    axes[1].plot(time, df["sen_neg"], color=neg_color, label=f"{sim_name}_neg")
    axes[1].set_ylabel(f"{sim_name} Neg [kPa]")

    if pos_active:
        axes[2].plot(time, df["dPpos_dt"], color=pos_color, label=f"{sim_name}_dPpos_dt")
    if neg_active:
        axes[2].plot(time, df["dPneg_dt"], color=neg_color, label=f"{sim_name}_dPneg_dt")
    if not pos_active and not neg_active:
        axes[2].plot(time, df["dPpos_dt"], color=pos_color, label=f"{sim_name}_dPpos_dt")
        axes[2].plot(time, df["dPneg_dt"], color=neg_color, label=f"{sim_name}_dPneg_dt")
    axes[2].set_ylabel(f"{sim_name} dP/dt")

    axes[3].plot(time, df["mean_pump_in"], color="purple", label=f"{sim_name}_pump_in")
    axes[3].plot(time, df["mean_pump_out"], color="brown", label=f"{sim_name}_pump_out")
    if pos_active:
        axes[3].plot(time, df["mean_valve_pos"], color=pos_color, label=f"{sim_name}_valve_pos")
    if neg_active:
        axes[3].plot(time, df["mean_valve_neg"], color=neg_color, label=f"{sim_name}_valve_neg")

    if show_valve_debug:
        if pos_active:
            axes[3].plot(
                time,
                df["pos_q_static_lpm"],
                color=pos_color,
                linestyle="--",
                alpha=0.65,
                label=f"{sim_name}_pos_q_static",
            )
            axes[3].plot(
                time,
                df["pos_q_pred_lpm"],
                color=pos_color,
                linestyle=":",
                alpha=0.9,
                label=f"{sim_name}_pos_q_pred",
            )
        if neg_active:
            axes[3].plot(
                time,
                df["neg_q_static_lpm"],
                color=neg_color,
                linestyle="--",
                alpha=0.65,
                label=f"{sim_name}_neg_q_static",
            )
            axes[3].plot(
                time,
                df["neg_q_pred_lpm"],
                color=neg_color,
                linestyle=":",
                alpha=0.9,
                label=f"{sim_name}_neg_q_pred",
            )
    axes[3].set_ylabel(f"{sim_name} Flow")

    axes[4].plot(time, df["ctrl_pos"], color=pos_color, label=f"{sim_name}_ctrl_pos")
    axes[4].plot(time, df["ctrl_neg"], color=neg_color, label=f"{sim_name}_ctrl_neg")
    axes[4].set_ylabel(f"{sim_name} Control")
    axes[4].set_xlabel("Time [sec]")

    for ax in axes:
        ax.grid(True)
        legend_above(ax, ncol=4)

    add_period_lines(list(axes), df, periods)
    fig.suptitle(f"{save_name} | {sim_name} | periods={periods}", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965], h_pad=2.0)
    fig.savefig(f"{out_dir}/{save_name}_{sim_name}.png", dpi=150)
    plt.close(fig)


def save_outputs(dfs: dict[str, pd.DataFrame], periods: list[float], args: argparse.Namespace) -> None:
    if args.save_name:
        save_name = args.save_name
    else:
        stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        sim_tag = "_".join(args.sims)
        save_name = f"{stamp}_{sim_tag}_bangbang"

    out_dir = f'{get_pkg_path("pneu_rl")}/exp/{save_name}'
    os.makedirs(out_dir, exist_ok=True)
    for sim_name, df in dfs.items():
        df.to_csv(f"{out_dir}/{save_name}_{sim_name}.csv", index=False)
        save_overview_plot(
            df,
            sim_name=sim_name,
            out_dir=out_dir,
            save_name=save_name,
            periods=periods,
            args=args,
        )
        if args.valve_debug:
            save_valve_debug_plot(
                df,
                sim_name=sim_name,
                out_dir=out_dir,
                save_name=save_name,
                periods=periods,
            )

    cfg = dict(
        bangbang=BANGBANG_CFG,
        args=vars(args),
        periods=periods,
        summary={
            sim_name: calc_summary(df)
            for sim_name, df in dfs.items()
        },
    )
    with open(f"{out_dir}/cfg.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"[INFO] Saved bang-bang result: {out_dir}")


def main() -> None:
    args = argparse.Namespace(
        freq=BANGBANG_CFG["freq"],
        delay=BANGBANG_CFG["delay"],
        scale=BANGBANG_CFG["scale"],
        sims=BANGBANG_CFG["sims"],
        periods=BANGBANG_CFG["periods"],
        start_period=BANGBANG_CFG["start_period"],
        end_period=BANGBANG_CFG["end_period"],
        period_ratio=BANGBANG_CFG["period_ratio"],
        cycles_per_period=BANGBANG_CFG["cycles_per_period"],
        valve_debug=BANGBANG_CFG["valve_debug"],
        save_name=BANGBANG_CFG["save_name"],
    )

    dfs, periods = run(args)
    save_outputs(dfs, periods, args)


if __name__ == "__main__":
    main()
