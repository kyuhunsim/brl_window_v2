import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pneu_env.sim3 import PneuSim
from pneu_ref.random_ref import RandomRef
from pneu_ref.sine_ref import SineRef
from pneu_ref.step_ref import StepRef
from pneu_utils.utils import get_pkg_path

try:
    from analyze_lib3_redundancy import analyze_csv, print_report
except ModuleNotFoundError:
    analyze_csv = None

    def print_report(report: dict) -> None:
        print(json.dumps(report, indent=2))


ATM = 101.325
VALVE_DEBUG_COLUMNS = [
    f"{valve}_{name}"
    for valve in (
        "ch_pos",
        "ch_neg",
        "act_pos_in",
        "act_pos_out",
        "act_neg_in",
        "act_neg_out",
    )
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

RANDOM_REF_KWARGS = dict(
    pos_min_off=115,
    pos_max_off=140,
    neg_min_off=60,
    neg_max_off=80,
    pos_max_ts=10,
    neg_max_ts=10,
    pos_max_amp=10,
    neg_max_amp=5,
    seed=61098,
)

SINE_REF_KWARGS = dict(
    pos_amp=10,
    pos_per=10,
    pos_off=130,
    neg_amp=5,
    neg_per=10,
    neg_off=70,
    iter=10,
)

STEP_REF_KWARGS = dict(
    time_step=5,
    ref_pos=np.array([115, 140, 125, 135, 120, 138], dtype=np.float64),
    ref_neg=np.array([60, 80, 70, 63, 75, 65], dtype=np.float64),
    extra_time=10,
)

BASELINE_CONFIG = dict(
    chamber=dict(
        mode="fixed",
        raw=0.0,
        margin=35.0,
        kp=0.04,
        ki=0.001,
        deadband=1.0,
        min_open=0.15,
        max_open=0.85,
        ch_pos_min=ATM,
        ch_pos_max=200.0,
        ch_neg_min=15.0,
        ch_neg_max=ATM,
    ),
    pid=dict(
        # Four actuator-side PID channels, matching pneu_env.pid.ActuatorPressurePID.
        act_pos_in=dict(kp=0.01, ki=0.004, kd=0.001),
        act_pos_out=dict(kp=0.05, ki=0.001, kd=0.001),
        act_neg_in=dict(kp=0.019, ki=0.0018, kd=0.001),
        act_neg_out=dict(kp=0.025, ki=0.0008, kd=0.001),
        deadband=0.5,
        integral_limit=100.0,
        max_open=1.0,
    ),
)


def get_pid_config(args: argparse.Namespace) -> dict:
    pid_cfg = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in BASELINE_CONFIG["pid"].items()
    }

    # Legacy shared-gain overrides. Prefer editing BASELINE_CONFIG above for
    # repeatable baseline experiments.
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
    if args.ctrl4_off:
        pid_cfg["act_pos_out"]["kp"] = 0.0
        pid_cfg["act_pos_out"]["ki"] = 0.0
        pid_cfg["act_pos_out"]["kd"] = 0.0

    return pid_cfg


def pid_gain_arrays(pid_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out")
    kp = np.array([pid_cfg[ch]["kp"] for ch in channels], dtype=np.float64)
    ki = np.array([pid_cfg[ch]["ki"] for ch in channels], dtype=np.float64)
    kd = np.array([pid_cfg[ch]["kd"] for ch in channels], dtype=np.float64)
    return kp, ki, kd


def make_ref(mode: str):
    if mode == "random":
        return RandomRef(**RANDOM_REF_KWARGS)
    if mode == "sine":
        return SineRef(**SINE_REF_KWARGS)
    if mode == "step":
        return StepRef(**STEP_REF_KWARGS)
    raise ValueError(f"Unknown ref mode: {mode}")


def make_sim(args: argparse.Namespace) -> PneuSim:
    sim = PneuSim(
        freq=args.freq,
        delay=args.delay,
        noise=False,
        noise_std=0,
        scale=args.scale,
    )
    sim.set_init_press(
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )
    return sim


def chamber_allocator(
    ch_press: np.ndarray,
    ref: np.ndarray,
    *,
    mode: str,
    chamber_raw: float,
    margin: float,
    kp: float,
    deadband: float,
    min_open: float,
    max_open: float,
    ch_pos_min: float,
    ch_pos_max: float,
    ch_neg_min: float,
    ch_neg_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    ctrl = np.array([chamber_raw, chamber_raw], dtype=np.float64)
    target = np.array([np.nan, np.nan], dtype=np.float64)
    if mode == "fixed":
        return ctrl, target
    if mode != "reserve":
        raise ValueError(f"Unknown chamber mode: {mode}")

    target[0] = np.clip(ref[0] + margin, ch_pos_min, ch_pos_max)
    target[1] = np.clip(ref[1] - margin, ch_neg_min, ch_neg_max)

    # Positive chamber valve opens toward lower pressure.
    pos_err = ch_press[0] - target[0]
    if pos_err > deadband:
        ctrl[0] = np.clip(pos_err * kp, min_open, max_open)
    else:
        ctrl[0] = 0.0

    # Negative chamber valve opens toward higher pressure.
    neg_err = target[1] - ch_press[1]
    if neg_err > deadband:
        ctrl[1] = np.clip(neg_err * kp, min_open, max_open)
    else:
        ctrl[1] = 0.0

    return ctrl, target


def conflict_free_allocator(
    act_press: np.ndarray,
    ref: np.ndarray,
    *,
    kp: float,
    deadband: float,
    min_open: float,
    max_open: float,
    chamber_ctrl: np.ndarray,
) -> np.ndarray:
    err = ref - act_press
    ctrl = np.zeros(6, dtype=np.float64)
    ctrl[0:2] = chamber_ctrl

    if abs(err[0]) > deadband:
        mag = np.clip(abs(err[0]) * kp, min_open, max_open)
        if err[0] > 0.0:
            ctrl[2] = mag
        else:
            ctrl[3] = mag

    if abs(err[1]) > deadband:
        mag = np.clip(abs(err[1]) * kp, min_open, max_open)
        if err[1] > 0.0:
            ctrl[4] = mag
        else:
            ctrl[5] = mag

    return ctrl


def pid_allocator(
    act_press: np.ndarray,
    ref: np.ndarray,
    prev_err: np.ndarray,
    integral_err: np.ndarray,
    *,
    dt: float,
    pid_cfg: dict,
    chamber_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = ref - act_press
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

    ctrl = np.zeros(6, dtype=np.float64)
    ctrl[0:2] = chamber_ctrl
    active = directional_err > float(pid_cfg["deadband"])
    ctrl[2:6] = np.where(
        active,
        np.clip(effort, 0.0, float(pid_cfg["max_open"])),
        0.0,
    )

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
        ctrl[0] = args.chamber_raw
        ctrl[1] = args.chamber_raw
        return ctrl, prev_err, integral_err, np.array([np.nan, np.nan])

    chamber_ctrl, chamber_target = chamber_allocator(
        ch_press,
        ref,
        mode=args.chamber_mode,
        chamber_raw=args.chamber_raw,
        margin=args.chamber_margin,
        kp=args.chamber_kp,
        deadband=args.chamber_deadband,
        min_open=args.chamber_min_open,
        max_open=args.chamber_max_open,
        ch_pos_min=args.ch_pos_min,
        ch_pos_max=args.ch_pos_max,
        ch_neg_min=args.ch_neg_min,
        ch_neg_max=args.ch_neg_max,
    )

    if mode == "bangbang":
        ctrl = conflict_free_allocator(
            act_press,
            ref,
            kp=args.bangbang_kp,
            deadband=(
                args.deadband
                if args.deadband is not None
                else BASELINE_CONFIG["pid"]["deadband"]
            ),
            min_open=args.min_open,
            max_open=(
                args.max_open
                if args.max_open is not None
                else BASELINE_CONFIG["pid"]["max_open"]
            ),
            chamber_ctrl=chamber_ctrl,
        )
        return ctrl, prev_err, integral_err, chamber_target
    if mode == "pid":
        ctrl, prev_err, integral_err = pid_allocator(
            act_press,
            ref,
            prev_err,
            integral_err,
            dt=dt,
            pid_cfg=get_pid_config(args),
            chamber_ctrl=chamber_ctrl,
        )
        return ctrl, prev_err, integral_err, chamber_target
    raise ValueError(f"Unknown controller mode: {mode}")


def run_controller(args: argparse.Namespace, mode: str) -> pd.DataFrame:
    sim = make_sim(args)
    ref_gen = make_ref(args.ref)

    rows = []
    curr_time = 0.0
    ch_press = np.array([ATM, ATM], dtype=np.float64)
    act_press = np.array([ATM, ATM], dtype=np.float64)
    prev_err = np.zeros(4, dtype=np.float64)
    integral_err = np.zeros(4, dtype=np.float64)
    dt = 1.0 / float(args.freq)

    while curr_time < args.max_time:
        ref = np.array(ref_gen.get_goal(curr_time), dtype=np.float64)
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
        valve_debug = sim.get_valve_debug() if args.valve_debug else {}

        curr_time = float(obs[0])
        ch_press = np.array([o["sen_pos"], o["sen_neg"]], dtype=np.float64)
        act_press = np.array([o["sen_act_pos"], o["sen_act_neg"]], dtype=np.float64)
        row = dict(
            curr_time=o["curr_time"],
            press_pos=o["sen_pos"],
            press_neg=o["sen_neg"],
            act_pos_press=o["sen_act_pos"],
            act_neg_press=o["sen_act_neg"],
            ref_act_pos=o["ref_act_pos"],
            ref_act_neg=o["ref_act_neg"],
            ref_ch_pos=float(chamber_target[0]),
            ref_ch_neg=float(chamber_target[1]),
            ctrl1=o["ctrl_pos"],
            ctrl2=o["ctrl_neg"],
            ctrl3=o["ctrl_act_pos_in"],
            ctrl4=o["ctrl_act_pos_out"],
            ctrl5=o["ctrl_act_neg_in"],
            ctrl6=o["ctrl_act_neg_out"],
            raw_ctrl1=float(ctrl[0]),
            raw_ctrl2=float(ctrl[1]),
            raw_ctrl3=float(ctrl[2]),
            raw_ctrl4=float(ctrl[3]),
            raw_ctrl5=float(ctrl[4]),
            raw_ctrl6=float(ctrl[5]),
            controller=mode,
        )
        if args.valve_debug:
            row.update({col: valve_debug.get(col, np.nan) for col in VALVE_DEBUG_COLUMNS})
        rows.append(row)

    return pd.DataFrame(rows)


def plot_controller(df: pd.DataFrame, mode: str, out_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    time = df["curr_time"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time, df["press_pos"], label="Chamber Pos", color="#d62728")
    axes[0].plot(time, df["press_neg"], label="Chamber Neg", color="#1f77b4")
    if "ref_ch_pos" in df.columns and not np.all(np.isnan(df["ref_ch_pos"])):
        axes[0].plot(time, df["ref_ch_pos"], label="Chamber Pos Ref", color="#d62728", linestyle="--")
        axes[0].plot(time, df["ref_ch_neg"], label="Chamber Neg Ref", color="#1f77b4", linestyle="--")
    axes[0].set_ylabel("Chamber [kPa]")
    axes[0].legend(loc="best")

    axes[1].plot(time, df["ref_act_pos"], label="Ref Pos", color="black", linestyle="--")
    axes[1].plot(time, df["act_pos_press"], label="Act Pos", color="#d62728")
    axes[1].plot(time, df["ref_act_neg"], label="Ref Neg", color="gray", linestyle="--")
    axes[1].plot(time, df["act_neg_press"], label="Act Neg", color="#1f77b4")
    axes[1].set_ylabel("Actuator [kPa]")
    axes[1].legend(loc="best", ncol=2)

    pos_common = 0.5 * (df["ctrl3"] + df["ctrl4"])
    pos_diff = df["ctrl3"] - df["ctrl4"]
    neg_common = 0.5 * (df["ctrl5"] + df["ctrl6"])
    neg_diff = df["ctrl6"] - df["ctrl5"]
    axes[2].plot(time, pos_common, label="Pos Common", color="#d62728")
    axes[2].plot(time, pos_diff, label="Pos Diff", color="#ff7f0e")
    axes[2].plot(time, neg_common, label="Neg Common", color="#1f77b4")
    axes[2].plot(time, neg_diff, label="Neg Diff", color="#9467bd")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Common / Diff")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="best", ncol=2)

    for ax in axes:
        ax.grid(True, color="0.85")
    fig.tight_layout()
    fig.savefig(out_dir / f"{mode}_main.png", dpi=150)

    ctrl_fig, ctrl_axes = plt.subplots(6, 1, figsize=(9, 10), sharex=True)
    ctrl_series = [
        ("ctrl1", "Positive Chamber", "#d62728"),
        ("ctrl2", "Negative Chamber", "#1f77b4"),
        ("ctrl3", "Actuator Positive In", "#2ca02c"),
        ("ctrl4", "Actuator Positive Out", "#ff7f0e"),
        ("ctrl5", "Actuator Negative In", "#9467bd"),
        ("ctrl6", "Actuator Negative Out", "#8c564b"),
    ]
    for ax, (key, title, color) in zip(ctrl_axes, ctrl_series):
        ax.plot(time, df[key], color=color, linewidth=1.8)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, color="0.85")
    ctrl_axes[-1].set_xlabel("Time [s]")
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

    if analyze_csv is None:
        pos_err = df["ref_act_pos"] - df["act_pos_press"]
        neg_err = df["ref_act_neg"] - df["act_neg_press"]
        report = dict(
            csv=str(csv_path),
            sample_count=int(len(df)),
            rmse_pos=float(np.sqrt(np.mean(pos_err * pos_err))),
            rmse_neg=float(np.sqrt(np.mean(neg_err * neg_err))),
            mae_pos=float(np.mean(np.abs(pos_err))),
            mae_neg=float(np.mean(np.abs(neg_err))),
        )
    else:
        report = analyze_csv(
            csv_path,
            saturation_threshold=args.sat_thr,
            raw_from_scaled=args.raw_from_scaled_report,
            scaled_low=0.85,
            scaled_high=1.0,
            window_start=args.window_start,
            window_end=args.window_end,
        )
    report["controller"] = mode
    print_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple lib3 controllers to separate simulator oscillation from RL action issues."
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["hold", "constant", "bangbang", "pid"],
        choices=["hold", "constant", "bangbang", "pid"],
    )
    parser.add_argument("--ref", choices=["random", "sine", "step"], default="random")
    parser.add_argument("--max-time", type=float, default=100.0)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--scale", action="store_true", default=True)
    parser.add_argument("--no-scale", dest="scale", action="store_false")
    parser.add_argument("--constant-raw", type=float, default=0.5)
    parser.add_argument("--chamber-raw", type=float, default=BASELINE_CONFIG["chamber"]["raw"])
    parser.add_argument("--chamber-mode", choices=["fixed", "reserve"], default=BASELINE_CONFIG["chamber"]["mode"])
    parser.add_argument("--chamber-margin", type=float, default=BASELINE_CONFIG["chamber"]["margin"])
    parser.add_argument("--chamber-kp", type=float, default=BASELINE_CONFIG["chamber"]["kp"])
    parser.add_argument("--chamber-deadband", type=float, default=BASELINE_CONFIG["chamber"]["deadband"])
    parser.add_argument("--chamber-min-open", type=float, default=BASELINE_CONFIG["chamber"]["min_open"])
    parser.add_argument("--chamber-max-open", type=float, default=BASELINE_CONFIG["chamber"]["max_open"])
    parser.add_argument("--ch-pos-min", type=float, default=BASELINE_CONFIG["chamber"]["ch_pos_min"])
    parser.add_argument("--ch-pos-max", type=float, default=BASELINE_CONFIG["chamber"]["ch_pos_max"])
    parser.add_argument("--ch-neg-min", type=float, default=BASELINE_CONFIG["chamber"]["ch_neg_min"])
    parser.add_argument("--ch-neg-max", type=float, default=BASELINE_CONFIG["chamber"]["ch_neg_max"])
    parser.add_argument("--deadband", type=float, default=None)
    parser.add_argument("--min-open", type=float, default=0.15)
    parser.add_argument("--max-open", type=float, default=None)
    parser.add_argument("--bangbang-kp", type=float, default=0.08)
    parser.add_argument("--pid-kp", type=float, default=None)
    parser.add_argument("--pid-ki", type=float, default=None)
    parser.add_argument("--pid-kd", type=float, default=None)
    parser.add_argument("--integral-limit", type=float, default=None)
    parser.add_argument("--ctrl4-off", action="store_true", help="Disable actuator positive out PID gains for diagnosis.")
    parser.add_argument("--sat-thr", type=float, default=0.95)
    parser.add_argument("--raw-from-scaled-report", action="store_true")
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--window-end", type=float, default=None)
    parser.add_argument("--valve-debug", action="store_true", default=True)
    parser.add_argument("--no-valve-debug", dest="valve_debug", action="store_false")
    parser.add_argument("--plot", action="store_true", help="Save baseline plots as PNG files.")
    parser.add_argument("--show", action="store_true", help="Show baseline plots interactively.")
    parser.add_argument("--save-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
    save_name = args.save_name or f"{timestamp}_lib3_baseline_diag_{args.ref}"
    out_dir = Path(get_pkg_path("pneu_rl")) / "exp" / save_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for mode in args.controllers:
        reports.append(save_and_analyze(args, mode, out_dir))

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            dict(
                args=vars(args),
                baseline_config=BASELINE_CONFIG,
                resolved_pid_config=get_pid_config(args),
                reports=reports,
            ),
            f,
            indent=2,
        )

    print(f"\n[INFO] saved lib3 baseline diagnostics: {out_dir}")


if __name__ == "__main__":
    main()
