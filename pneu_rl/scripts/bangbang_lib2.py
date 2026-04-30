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
    save_name = None,
)


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

            rows.append(
                dict(
                    curr_time=o["curr_time"],
                    period=period,
                    sen_pos=o["sen_pos"],
                    sen_neg=o["sen_neg"],
                    action_pos=float(action[0]),
                    action_neg=float(action[1]),
                    ctrl_pos=o["ctrl_pos"],
                    ctrl_neg=o["ctrl_neg"],
                )
            )

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
        ctrl_pos_min=float(df["ctrl_pos"].min()),
        ctrl_pos_max=float(df["ctrl_pos"].max()),
        ctrl_neg_min=float(df["ctrl_neg"].min()),
        ctrl_neg_max=float(df["ctrl_neg"].max()),
    )


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

    fig = plt.figure(figsize=(12, max(8, 4 * len(dfs))))
    gs = gridspec.GridSpec(3 * len(dfs), 1, figure=fig)

    first_ax = None
    axes = []
    colors = dict(
        sim2=("red", "blue"),
        sim=("darkorange", "deepskyblue"),
    )

    for idx, (sim_name, df) in enumerate(dfs.items()):
        base_idx = 3 * idx
        if first_ax is None:
            ax1 = fig.add_subplot(gs[base_idx, 0])
            first_ax = ax1
        else:
            ax1 = fig.add_subplot(gs[base_idx, 0], sharex=first_ax)
        ax2 = fig.add_subplot(gs[base_idx + 1, 0], sharex=first_ax)
        ax3 = fig.add_subplot(gs[base_idx + 2, 0], sharex=first_ax)
        axes.extend([ax1, ax2, ax3])

        pos_color, neg_color = colors[sim_name]

        ax1.plot(df["curr_time"], df["sen_pos"], color=pos_color, label=f"{sim_name}_pos")
        ax1.grid(True)
        ax1.legend(loc="upper right")
        ax1.set_ylabel(f"{sim_name} Pos [kPa]")

        ax2.plot(df["curr_time"], df["sen_neg"], color=neg_color, label=f"{sim_name}_neg")
        ax2.grid(True)
        ax2.legend(loc="upper right")
        ax2.set_ylabel(f"{sim_name} Neg [kPa]")

        ax3.plot(df["curr_time"], df["ctrl_pos"], color=pos_color, label=f"{sim_name}_ctrl_pos")
        ax3.plot(df["curr_time"], df["ctrl_neg"], color=neg_color, label=f"{sim_name}_ctrl_neg")
        ax3.grid(True)
        ax3.legend(loc="upper right")
        ax3.set_ylabel(f"{sim_name} Control")
        if idx == len(dfs) - 1:
            ax3.set_xlabel("Time [sec]")

    for period in periods[1:]:
        idx = df.index[df["period"] == period]
        if len(idx) == 0:
            continue
        t0 = float(df.loc[idx[0], "curr_time"])
        for ax in axes:
            ax.axvline(t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.suptitle(
        f"{save_name} | periods={periods}"
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{save_name}.png", dpi=150)
    plt.close(fig)

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
        save_name=BANGBANG_CFG["save_name"],
    )

    dfs, periods = run(args)
    save_outputs(dfs, periods, args)


if __name__ == "__main__":
    main()
