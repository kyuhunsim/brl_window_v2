from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim4 import PneuSim
from pneu_ref.disp_ref import PressureDiffDisplacementRef
from pneu_ref.random_ref import RandomRef
from pneu_ref.sine_ref import SineRef
from pneu_ref.step_ref import StepCasesRef
from pneu_utils.utils import get_pkg_path, load_yaml


LIB4_TUNED_PARAMS_PATH = (
    Path(get_pkg_path("pneu_env"))
    / "coeff_result"
    / "260624_11_21_35_tuner4_output48plus49_linit34807_leak_tuned"
    / "params.yaml"
)

BASELINE_CONFIG = dict(
    model_cfg="0625_lib4_Ours",
    chamber=dict(
        mode="fixed",
        raw_pos=0.5,
        raw_neg=1.0,
        margin_pos=25.0,
        margin_neg=55.0,
        kp=0.04,
        deadband=1.0,
        min_open=0.15,
        max_open=0.85,
        pos_min=110.0,
        pos_max=180.0,
        neg_min=15.0,
        neg_max=45.0,
    ),
    pid=dict(
        act_pos_in  = dict(kp=0.1, ki=0.0, kd=0.0),
        act_pos_out = dict(kp=0.1, ki=0.0, kd=0.0),
        act_neg_in  = dict(kp=0.1, ki=0.0, kd=0.0),
        act_neg_out = dict(kp=0.1, ki=0.0, kd=0.0),
        closed_signal=0.0,
        open_span=1.0,
        deadband=0.5,
        integral_limit=100.0,
        max_open=1.0,
    ),
)


class TrajRefND:
    def __init__(self, traj_time, traj_values):
        self.traj_time = np.asarray(traj_time, dtype=np.float64)
        self.traj_values = np.asarray(traj_values, dtype=np.float64)
        self.max_time = float(self.traj_time[-1]) if len(self.traj_time) else 0.0

    def get_goal(self, curr_time):
        if curr_time <= self.traj_time[0]:
            return tuple(self.traj_values[0])
        if curr_time >= self.traj_time[-1]:
            return tuple(self.traj_values[-1])
        return tuple(
            np.interp(curr_time, self.traj_time, self.traj_values[:, idx])
            for idx in range(self.traj_values.shape[1])
        )


def load_lib4_tuned_params() -> dict[str, float]:
    with LIB4_TUNED_PARAMS_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return dict(
        pump_in_coeff=float(raw["pump_in_coeff"]),
        pump_out_coeff=float(raw["pump_out_coeff"]),
        leak_pos_atm=float(raw["leak_pos_atm"]),
        leak_neg_atm=float(raw["leak_neg_atm"]),
        leak_cross=float(raw["leak_cross"]),
    )


def get_pid_config(args: argparse.Namespace) -> dict:
    pid_cfg = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in BASELINE_CONFIG["pid"].items()
    }
    if args.pid_kp is not None:
        for key in ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
            pid_cfg[key]["kp"] = args.pid_kp
    if args.pid_ki is not None:
        for key in ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
            pid_cfg[key]["ki"] = args.pid_ki
    if args.pid_kd is not None:
        for key in ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
            pid_cfg[key]["kd"] = args.pid_kd
    if args.deadband is not None:
        pid_cfg["deadband"] = args.deadband
    if args.integral_limit is not None:
        pid_cfg["integral_limit"] = args.integral_limit
    if args.max_open is not None:
        pid_cfg["max_open"] = args.max_open
    return pid_cfg


def pid_gain_arrays(pid_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out")
    kp = np.array([pid_cfg[ch]["kp"] for ch in channels], dtype=np.float64)
    ki = np.array([pid_cfg[ch]["ki"] for ch in channels], dtype=np.float64)
    kd = np.array([pid_cfg[ch]["kd"] for ch in channels], dtype=np.float64)
    return kp, ki, kd


def make_ref(args: argparse.Namespace, cfg: dict):
    if args.ref == "random":
        pressure_ref = RandomRef(**cfg["rnd_ref"])
    elif args.ref == "sine":
        pressure_ref = SineRef(
            pos_amp=4,
            pos_per=10,
            pos_off=120,
            neg_amp=2,
            neg_per=10,
            neg_off=86,
            iter=4,
        )
    elif args.ref == "step":
        pressure_ref = StepCasesRef(
            time_step=5,
            ref_pos_max=126,
            ref_pos_min=112,
            ref_neg_max=88,
            ref_neg_min=84,
        )
    elif args.ref == "trajectory":
        if args.traj_csv is None:
            raise ValueError("--traj-csv is required for --ref trajectory")
        df = pd.read_csv(args.traj_csv)
        t = df["curr_time"].to_numpy(dtype=np.float64)
        t = t - t[0]
        pos = df["ref_act_pos"].to_numpy(dtype=np.float64)
        neg = df["ref_act_neg"].to_numpy(dtype=np.float64)
        if "ref_displacement" in df:
            disp = df["ref_displacement"].to_numpy(dtype=np.float64)
            return TrajRefND(t, np.c_[pos, neg, disp])
        pressure_ref = TrajRefND(t, np.c_[pos, neg])
    else:
        raise ValueError(f"Unknown ref mode: {args.ref}")

    return PressureDiffDisplacementRef(
        pressure_ref,
        **cfg["pressure_diff_disp_ref"],
    )


def make_sim(args: argparse.Namespace, cfg: dict) -> PneuSim:
    sim_kwargs = dict(cfg["obs"])
    sim_kwargs.update(freq=args.freq, delay=args.delay, noise=False, scale=args.scale)
    sim = PneuSim(**sim_kwargs)

    tuned = load_lib4_tuned_params()
    sim.set_discharge_coeff(
        inlet_pump_coeff=tuned["pump_in_coeff"],
        outlet_pump_coeff=tuned["pump_out_coeff"],
    )
    sim.set_leak_coefficients(
        pos_atm=tuned["leak_pos_atm"],
        neg_atm=tuned["leak_neg_atm"],
        cross=tuned["leak_cross"],
    )
    return sim


def chamber_allocator(
    ch_press: np.ndarray,
    ref: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    ctrl = np.array([args.chamber_raw_pos, args.chamber_raw_neg], dtype=np.float64)
    target = np.array([np.nan, np.nan], dtype=np.float64)
    if args.chamber_mode == "fixed":
        return ctrl, target
    if args.chamber_mode != "reserve":
        raise ValueError(f"Unknown chamber mode: {args.chamber_mode}")

    target[0] = np.clip(ref[0] + args.chamber_margin_pos, args.ch_pos_min, args.ch_pos_max)
    target[1] = np.clip(ref[1] - args.chamber_margin_neg, args.ch_neg_min, args.ch_neg_max)

    pos_err = ch_press[0] - target[0]
    if pos_err > args.chamber_deadband:
        ctrl[0] = np.clip(pos_err * args.chamber_kp, args.chamber_min_open, args.chamber_max_open)
    else:
        ctrl[0] = 0.0

    neg_err = target[1] - ch_press[1]
    if neg_err > args.chamber_deadband:
        ctrl[1] = np.clip(neg_err * args.chamber_kp, args.chamber_min_open, args.chamber_max_open)
    else:
        ctrl[1] = 0.0

    return ctrl, target


def pid_allocator(
    act_press: np.ndarray,
    ref: np.ndarray,
    prev_err: np.ndarray,
    integral_err: np.ndarray,
    *,
    args: argparse.Namespace,
    chamber_ctrl: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pid_cfg = get_pid_config(args)
    err = ref[:2] - act_press
    directional_err = np.array(
        [
            max(err[0], 0.0),
            max(-err[0], 0.0),
            max(err[1], 0.0),
            max(-err[1], 0.0),
        ],
        dtype=np.float64,
    )
    derr = (directional_err - prev_err) / max(dt, 1e-9)
    integral_err = np.clip(
        integral_err + directional_err * dt,
        -float(pid_cfg["integral_limit"]),
        float(pid_cfg["integral_limit"]),
    )
    kp, ki, kd = pid_gain_arrays(pid_cfg)
    effort = kp * directional_err + ki * integral_err + kd * derr

    closed_signal = float(pid_cfg["closed_signal"])
    open_span = float(pid_cfg["open_span"])
    ctrl = np.full(6, closed_signal, dtype=np.float64)
    ctrl[0:2] = chamber_ctrl
    active = directional_err > float(pid_cfg["deadband"])
    open_ratio = np.clip(effort, 0.0, float(pid_cfg["max_open"]))
    ctrl[2:6] = np.where(active, closed_signal + open_span * open_ratio, closed_signal)
    return ctrl, directional_err, integral_err


def controller_action(
    mode: str,
    *,
    args: argparse.Namespace,
    ch_press: np.ndarray,
    act_press: np.ndarray,
    ref: np.ndarray,
    prev_err: np.ndarray,
    integral_err: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mode == "constant":
        return np.full(6, args.constant_raw, dtype=np.float64), prev_err, integral_err, np.array([np.nan, np.nan])
    if mode == "hold":
        ctrl = np.zeros(6, dtype=np.float64)
        ctrl[0] = args.chamber_raw_pos
        ctrl[1] = args.chamber_raw_neg
        return ctrl, prev_err, integral_err, np.array([np.nan, np.nan])

    chamber_ctrl, chamber_target = chamber_allocator(ch_press, ref, args=args)
    if mode == "pid":
        ctrl, prev_err, integral_err = pid_allocator(
            act_press,
            ref,
            prev_err,
            integral_err,
            args=args,
            chamber_ctrl=chamber_ctrl,
            dt=dt,
        )
        return ctrl, prev_err, integral_err, chamber_target
    raise ValueError(f"Unknown controller mode: {mode}")


def run_controller(args: argparse.Namespace, mode: str) -> pd.DataFrame:
    cfg = load_yaml(args.model_cfg)
    sim = make_sim(args, cfg)
    ref_gen = make_ref(args, cfg)

    rows = []
    curr_time = 0.0
    dt = 1.0 / float(args.freq)
    ch_press = np.array([sim.init_pos_press, sim.init_neg_press], dtype=np.float64)
    act_press = np.array([sim.init_act_pos_press, sim.init_act_neg_press], dtype=np.float64)
    prev_err = np.zeros(4, dtype=np.float64)
    integral_err = np.zeros(4, dtype=np.float64)

    while curr_time < args.max_time:
        ref = np.asarray(ref_gen.get_goal(curr_time), dtype=np.float64)
        ctrl, prev_err, integral_err, chamber_target = controller_action(
            mode,
            args=args,
            ch_press=ch_press,
            act_press=act_press,
            ref=ref,
            prev_err=prev_err,
            integral_err=integral_err,
            dt=dt,
        )
        obs, info = sim.observe(ctrl, ref)
        o = info["Observation"]
        curr_time = float(o["curr_time"])
        ch_press = np.array([o["sen_pos"], o["sen_neg"]], dtype=np.float64)
        act_press = np.array([o["sen_act_pos"], o["sen_act_neg"]], dtype=np.float64)
        env_disp = sim.total_length_to_contraction(o["sen_total_length"]) * 1000.0
        env_vel = -o["sen_total_velocity"] * 1000.0

        rows.append(
            dict(
                curr_time=curr_time,
                press_pos=o["sen_pos"],
                press_neg=o["sen_neg"],
                act_pos_press=o["sen_act_pos"],
                act_neg_press=o["sen_act_neg"],
                ref_act_pos=o["ref_act_pos"],
                ref_act_neg=o["ref_act_neg"],
                displacement=env_disp,
                displacement_vel=env_vel,
                ref_displacement=o.get("ref_displacement", np.nan),
                ref_ch_pos=float(chamber_target[0]),
                ref_ch_neg=float(chamber_target[1]),
                ctrl_ch_pos=o["ctrl_pos"],
                ctrl_ch_neg=o["ctrl_neg"],
                ctrl_act_pos_in=o["ctrl_act_pos_in"],
                ctrl_act_pos_out=o["ctrl_act_pos_out"],
                ctrl_act_neg_in=o["ctrl_act_neg_in"],
                ctrl_act_neg_out=o["ctrl_act_neg_out"],
                raw_ctrl_ch_pos=float(ctrl[0]),
                raw_ctrl_ch_neg=float(ctrl[1]),
                raw_ctrl_act_pos_in=float(ctrl[2]),
                raw_ctrl_act_pos_out=float(ctrl[3]),
                raw_ctrl_act_neg_in=float(ctrl[4]),
                raw_ctrl_act_neg_out=float(ctrl[5]),
                controller=mode,
            )
        )

    return pd.DataFrame(rows)


def analyze_df(df: pd.DataFrame) -> dict[str, float]:
    report = {}
    for name, ref_col, obs_col in (
        ("pos", "ref_act_pos", "act_pos_press"),
        ("neg", "ref_act_neg", "act_neg_press"),
        ("disp", "ref_displacement", "displacement"),
    ):
        err = df[ref_col].to_numpy(dtype=np.float64) - df[obs_col].to_numpy(dtype=np.float64)
        report[f"{name}_rmse"] = float(np.sqrt(np.mean(err * err)))
        report[f"{name}_mae"] = float(np.mean(np.abs(err)))
        report[f"{name}_bias"] = float(np.mean(err))
    return report


def plot_controller(df: pd.DataFrame, mode: str, out_dir: Path, show: bool) -> None:
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fontname = None
    if Path(font_path).is_file():
        import matplotlib.font_manager as fm

        fontname = fm.FontProperties(fname=font_path)
    label_font_size = 18
    time = df["curr_time"].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(8, 1, figure=fig)
    axes = [
        fig.add_subplot(gs[0:2, 0]),
        fig.add_subplot(gs[2:4, 0]),
        fig.add_subplot(gs[4:6, 0]),
        fig.add_subplot(gs[6:8, 0]),
    ]

    axes[0].plot(time, df["press_pos"], linewidth=2, color="red", label="CH_POS")
    axes[0].plot(time, df["press_neg"], linewidth=2, color="blue", label="CH_NEG")
    axes[0].set_ylabel("Chamber [kPa]", fontproperties=fontname, fontsize=label_font_size)

    axes[1].plot(time, df["ref_act_pos"], linewidth=2, color="black", label="REF")
    axes[1].plot(time, df["act_pos_press"], linewidth=2, color="red", label="ACT_POS")
    axes[1].set_ylabel("Act Pos [kPa]", fontproperties=fontname, fontsize=label_font_size)

    axes[2].plot(time, df["ref_act_neg"], linewidth=2, color="black", label="REF")
    axes[2].plot(time, df["act_neg_press"], linewidth=2, color="blue", label="ACT_NEG")
    axes[2].set_ylabel("Act Neg [kPa]", fontproperties=fontname, fontsize=label_font_size)

    axes[3].plot(time, df["ref_displacement"], linewidth=2, color="black", label="REF")
    axes[3].plot(time, df["displacement"], linewidth=2, color="green", label="DISP")
    axes[3].set_ylabel("Displacement [mm]", fontproperties=fontname, fontsize=label_font_size)
    axes[3].set_xlabel("Time [sec]", fontproperties=fontname, fontsize=label_font_size)

    for ax in axes:
        ax.grid(which="major", color="silver", linewidth=1)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)
        ax.minorticks_on()
        ax.legend(loc="upper right")
        ax.set(xlim=(0, None), ylim=(None, None))
    fig.tight_layout()
    fig.savefig(out_dir / f"{mode}_main.png", dpi=150)

    ctrl_fig, ctrl_axes = plt.subplots(6, 1, figsize=(9, 10), sharex=True)
    ctrl_series = [
        ("ctrl_ch_pos", "Positive Chamber", "#d62728"),
        ("ctrl_ch_neg", "Negative Chamber", "#1f77b4"),
        ("ctrl_act_pos_in", "Actuator Positive In", "#2ca02c"),
        ("ctrl_act_pos_out", "Actuator Positive Out", "#ff7f0e"),
        ("ctrl_act_neg_in", "Actuator Negative In", "#9467bd"),
        ("ctrl_act_neg_out", "Actuator Negative Out", "#8c564b"),
    ]
    for ax, (key, title, color_code) in zip(ctrl_axes, ctrl_series):
        ax.plot(time, df[key], linewidth=1.8, color=color_code)
        ax.set_ylabel(title, fontproperties=fontname, fontsize=12)
        ax.grid(which="major", color="silver", linewidth=1)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)
        ax.minorticks_on()
        ax.set(xlim=(0, None), ylim=(None, None))
    ctrl_axes[-1].set_xlabel("Time [sec]", fontproperties=fontname, fontsize=label_font_size)
    ctrl_fig.tight_layout()
    ctrl_fig.savefig(out_dir / f"{mode}_ctrl.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(ctrl_fig)


def save_and_analyze(args: argparse.Namespace, mode: str, out_dir: Path) -> dict:
    df = run_controller(args, mode)
    csv_path = out_dir / f"{mode}.csv"
    df.to_csv(csv_path, index=False)
    if args.plot or args.show:
        plot_controller(df, mode, out_dir, show=args.show)
    report = analyze_df(df)
    report["controller"] = mode
    report["csv"] = str(csv_path)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lib4 baseline controllers in the baseline_lib2/3 style."
    )
    parser.add_argument("--controllers", nargs="+", default=["pid"], choices=["hold", "constant", "pid"])
    parser.add_argument("--model-cfg", default=BASELINE_CONFIG["model_cfg"])
    parser.add_argument("--ref", choices=["random", "sine", "step", "trajectory"], default="random")
    parser.add_argument("--traj-csv", type=str, default=None)
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--scale", action="store_true", default=True)
    parser.add_argument("--no-scale", dest="scale", action="store_false")
    parser.add_argument("--constant-raw", type=float, default=0.5)
    parser.add_argument("--chamber-mode", choices=["fixed", "reserve"], default=BASELINE_CONFIG["chamber"]["mode"])
    parser.add_argument("--chamber-raw-pos", type=float, default=BASELINE_CONFIG["chamber"]["raw_pos"])
    parser.add_argument("--chamber-raw-neg", type=float, default=BASELINE_CONFIG["chamber"]["raw_neg"])
    parser.add_argument("--chamber-margin-pos", type=float, default=BASELINE_CONFIG["chamber"]["margin_pos"])
    parser.add_argument("--chamber-margin-neg", type=float, default=BASELINE_CONFIG["chamber"]["margin_neg"])
    parser.add_argument("--chamber-kp", type=float, default=BASELINE_CONFIG["chamber"]["kp"])
    parser.add_argument("--chamber-deadband", type=float, default=BASELINE_CONFIG["chamber"]["deadband"])
    parser.add_argument("--chamber-min-open", type=float, default=BASELINE_CONFIG["chamber"]["min_open"])
    parser.add_argument("--chamber-max-open", type=float, default=BASELINE_CONFIG["chamber"]["max_open"])
    parser.add_argument("--ch-pos-min", type=float, default=BASELINE_CONFIG["chamber"]["pos_min"])
    parser.add_argument("--ch-pos-max", type=float, default=BASELINE_CONFIG["chamber"]["pos_max"])
    parser.add_argument("--ch-neg-min", type=float, default=BASELINE_CONFIG["chamber"]["neg_min"])
    parser.add_argument("--ch-neg-max", type=float, default=BASELINE_CONFIG["chamber"]["neg_max"])
    parser.add_argument("--pid-kp", type=float, default=None)
    parser.add_argument("--pid-ki", type=float, default=None)
    parser.add_argument("--pid-kd", type=float, default=None)
    parser.add_argument("--deadband", type=float, default=None)
    parser.add_argument("--integral-limit", type=float, default=None)
    parser.add_argument("--max-open", type=float, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_name = args.save_name or datetime.now().strftime("%y%m%d_%H_%M_%S_lib4_baseline")
    out_dir = Path(get_pkg_path("pneu_rl")) / "exp" / save_name
    out_dir.mkdir(parents=True, exist_ok=False)

    reports = {}
    for mode in args.controllers:
        reports[mode] = save_and_analyze(args, mode, out_dir)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), reports=reports), f, indent=2)
    print(f"[ INFO] Saved lib4 baseline: {out_dir}")


if __name__ == "__main__":
    main()
