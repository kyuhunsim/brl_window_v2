import argparse
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

from pneu_env.env import PneuEnv
from pneu_env.pred2 import PneuPred
from pneu_env.sim import PneuSim as PneuSimBase
from pneu_env.sim2 import PneuSim as PneuSim2
from pneu_ref.random_ref import RandomRef
from pneu_ref.sine_ref import SineRef
from pneu_ref.step_ref import StepRef
from pneu_rl.sac import SAC
from pneu_utils.utils import get_pkg_path


ATM = 101.325

RANDOM_REF_KWARGS = dict(
    pos_max_off = 200,
    pos_min_off = 150,
    neg_max_off = 30,
    neg_min_off = 15,
    pos_max_ts = 10,
    neg_max_ts = 10,
    pos_max_amp = 10,
    neg_max_amp = 10,
    seed = 61098,
)

SINE_REF_KWARGS = dict(
    pos_amp = 10,
    pos_per = 5,
    pos_off = 80 + ATM,
    neg_amp = 7,
    neg_per = 7,
    neg_off = -70 + ATM,
    iter = 20,
    buf_time = 0,
)

COS_REF_KWARGS = dict(
    pos_amp = 10,
    pos_per = 10,
    pos_off = 80 + ATM,
    neg_amp = 7,
    neg_per = 10,
    neg_off = -80 + ATM,
    iter = 20,
    buf_time = 0,
)

STEP_REF_KWARGS = dict(
    time_step = 3,
    max_time = 100,
    pos_min = 150,
    pos_max = 200,
    neg_min = 10,
    neg_max = 30,
    seed = 61098,
    extra_time = 10,
)

REF_KWARGS = dict(
    random = RANDOM_REF_KWARGS,
    sin = SINE_REF_KWARGS,
    cos = COS_REF_KWARGS,
    step = STEP_REF_KWARGS,
)

PID_GAINS = dict(
    pos = dict(
        p = 0.1,
        i = 0.0015,
        d = 0.0000,
        integral_limit = 200.0,
    ),
    neg = dict(
        p = 0.1,
        i = 0.01,
        d = 0.0000,
        integral_limit = 200.0,
    ),
)


class CosRef(SineRef):
    def get_sin_value(
        self,
        x: float,
        amp: float,
        per: float,
        off: float,
    ) -> np.ndarray:
        return amp * np.cos(2 * np.pi * x / per) + off


def make_ref(ref_mode: str):
    if ref_mode == "random":
        return RandomRef(**RANDOM_REF_KWARGS)
    if ref_mode == "sin":
        return SineRef(**SINE_REF_KWARGS)
    if ref_mode == "cos":
        return CosRef(**COS_REF_KWARGS)
    if ref_mode == "step":
        rng = np.random.default_rng(STEP_REF_KWARGS["seed"])
        num_steps = int(np.ceil(STEP_REF_KWARGS["max_time"] / STEP_REF_KWARGS["time_step"]))
        ref_pos = rng.integers(
            STEP_REF_KWARGS["pos_min"],
            STEP_REF_KWARGS["pos_max"] + 1,
            size=num_steps,
        ).astype(float)
        ref_neg = rng.integers(
            STEP_REF_KWARGS["neg_min"],
            STEP_REF_KWARGS["neg_max"] + 1,
            size=num_steps,
        ).astype(float)
        return StepRef(
            time_step=STEP_REF_KWARGS["time_step"],
            ref_pos=ref_pos,
            ref_neg=ref_neg,
            extra_time=STEP_REF_KWARGS["extra_time"],
        )
    raise ValueError(f"Unknown ref mode: {ref_mode}")


def calc_metrics(
    df: pd.DataFrame,
    *,
    sen_pos_col: str = "sen_pos",
    sen_neg_col: str = "sen_neg",
) -> dict:
    pos_err = df["ref_pos"].to_numpy(dtype=np.float64) - df[sen_pos_col].to_numpy(dtype=np.float64)
    neg_err = df["ref_neg"].to_numpy(dtype=np.float64) - df[sen_neg_col].to_numpy(dtype=np.float64)
    return dict(
        rmse_pos = float(np.sqrt(np.mean(pos_err * pos_err))),
        rmse_neg = float(np.sqrt(np.mean(neg_err * neg_err))),
        mae_pos = float(np.mean(np.abs(pos_err))),
        mae_neg = float(np.mean(np.abs(neg_err))),
    )


def pid_ctrl(
    obs: np.ndarray,
    ref: np.ndarray,
    prev_err: np.ndarray,
    integral_err: np.ndarray,
    gains: dict,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = ref - obs
    derr = (err - prev_err) / max(dt, 1e-9)

    integral_err = integral_err + err * dt
    integral_err[0] = np.clip(
        integral_err[0],
        -gains["pos"]["integral_limit"],
        gains["pos"]["integral_limit"],
    )
    integral_err[1] = np.clip(
        integral_err[1],
        -gains["neg"]["integral_limit"],
        gains["neg"]["integral_limit"],
    )

    u_pos = -(
        gains["pos"]["p"] * err[0]
        + gains["pos"]["i"] * integral_err[0]
        + gains["pos"]["d"] * derr[0]
    )
    u_neg = (
        gains["neg"]["p"] * err[1]
        + gains["neg"]["i"] * integral_err[1]
        + gains["neg"]["d"] * derr[1]
    )

    action = np.array([u_pos, u_neg], dtype=np.float64)
    return np.clip(action, -1.0, 1.0), err, integral_err


def make_simulator(args: argparse.Namespace, sim_kind: str):
    if sim_kind == "sim2":
        sim_cls = PneuSim2
    elif sim_kind == "sim":
        sim_cls = PneuSimBase
    else:
        raise ValueError(f"Unknown sim kind: {sim_kind}")

    sim = sim_cls(
        freq=args.freq,
        delay=args.delay,
        noise=False,
        scale=args.scale,
    )
    sim.set_init_press(ATM, ATM)
    return sim


def run_pid(args: argparse.Namespace, sim_kind: str = "sim2") -> pd.DataFrame:
    sim = make_simulator(args, sim_kind)

    ref = make_ref(args.ref)

    rows = []
    obs = np.array([ATM, ATM], dtype=np.float64)
    prev_err = np.zeros(2, dtype=np.float64)
    integral_err = np.zeros(2, dtype=np.float64)
    curr_time = 0.0
    dt = 1.0 / float(args.freq)

    while curr_time < args.max_time:
        ref_pos, ref_neg = ref.get_goal(curr_time)
        goal = np.array([ref_pos, ref_neg], dtype=np.float64)

        action, prev_err, integral_err = pid_ctrl(
            obs,
            goal,
            prev_err,
            integral_err,
            PID_GAINS,
            dt=dt,
        )

        next_obs, info = sim.observe(action, goal)
        curr_time = float(next_obs[0])
        obs = next_obs[1:3].astype(np.float64)
        o = info["Observation"]
        rows.append(
            dict(
                curr_time=o["curr_time"],
                sen_pos=o["sen_pos"],
                sen_neg=o["sen_neg"],
                ref_pos=o["ref_pos"],
                ref_neg=o["ref_neg"],
                action_pos=float(action[0]),
                action_neg=float(action[1]),
                ctrl_pos=o["ctrl_pos"],
                ctrl_neg=o["ctrl_neg"],
            )
        )

    return pd.DataFrame(rows)


def load_model_cfg(model_name: str) -> dict:
    cfg_path = f'{get_pkg_path("pneu_rl")}/models/{model_name}/cfg.yaml'
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def run_rl(args: argparse.Namespace) -> pd.DataFrame:
    cfg = load_model_cfg(args.model_name)

    obs = PneuSim2(**cfg["obs"])
    pred = PneuPred(**cfg["pred"]) if cfg.get("pred") is not None else None
    obs.set_init_press(ATM, ATM)

    env = PneuEnv(
        obs=obs,
        ref=make_ref(args.ref),
        pred=pred,
        **cfg["env"],
    )
    env.verbose = lambda info: None
    model = SAC(
        env=env,
        **cfg["model"],
    )
    model.load(name=args.model_name)

    rows = []
    state, info = env.reset()
    curr_time = 0.0

    while curr_time < args.max_time:
        action = model.predict(state)
        state, _, _, _, info = env.step(action)
        o = info["obs"]
        curr_time = o["curr_time"]
        rows.append(
            dict(
                curr_time=o["curr_time"],
                sen_pos=o["sen_pos"],
                sen_neg=o["sen_neg"],
                ref_pos=o["ref_pos"],
                ref_neg=o["ref_neg"],
                action_pos=float(action[0]),
                action_neg=float(action[1]),
                ctrl_pos=o["ctrl_pos"],
                ctrl_neg=o["ctrl_neg"],
            )
        )

    env.close()
    return pd.DataFrame(rows)


RUN_COLORS = dict(
    pid_sim2_pos="red",
    pid_sim2_neg="blue",
    pid_sim_pos="darkorange",
    pid_sim_neg="deepskyblue",
    rl_sim2_pos="green",
    rl_sim2_neg="purple",
)


def run_selected(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if args.runs is not None:
        run_names = args.runs
    elif args.plot_mode == "pid_rl":
        run_names = ["pid_sim2", "rl_sim2"]
    else:
        run_names = ["pid_sim2"]

    dfs = {}
    for run_name in run_names:
        if run_name == "pid_sim2":
            dfs[run_name] = run_pid(args, sim_kind="sim2")
        elif run_name == "pid_sim":
            dfs[run_name] = run_pid(args, sim_kind="sim")
        elif run_name == "rl_sim2":
            dfs[run_name] = run_rl(args)
        else:
            raise ValueError(f"Unknown run: {run_name}")

    return dfs


def save_outputs(run_dfs: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    metrics = {
        run_name: calc_metrics(df)
        for run_name, df in run_dfs.items()
    }
    if args.save_name:
        save_name = args.save_name
    else:
        stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        run_tag = "_".join(run_dfs.keys())
        save_name = f"{stamp}_lib2_{args.ref}_{run_tag}_baseline"

    out_dir = f'{get_pkg_path("pneu_rl")}/exp/{save_name}'
    os.makedirs(out_dir, exist_ok=True)
    for run_name, df in run_dfs.items():
        df.to_csv(f"{out_dir}/{save_name}_{run_name}.csv", index=False)

    cfg = dict(
        plot_mode=args.plot_mode,
        runs=list(run_dfs.keys()),
        model_name=args.model_name if "rl_sim2" in run_dfs else None,
        sim=dict(freq=args.freq, delay=args.delay, scale=args.scale),
        ref=dict(mode=args.ref, kwargs=REF_KWARGS[args.ref]),
        pid_gains=PID_GAINS,
        controller=vars(args),
        metrics=metrics,
    )
    with open(f"{out_dir}/cfg.yaml", "w") as f:
        yaml.dump(cfg, f)

    num_plots = 3 * len(run_dfs)
    fig_height = max(7, 4 * len(run_dfs))
    fig = plt.figure(figsize=(12, fig_height))
    gs = gridspec.GridSpec(num_plots, 1, figure=fig)

    first_ax = None
    for idx, (run_name, df) in enumerate(run_dfs.items()):
        base_idx = 3 * idx
        if first_ax is None:
            ax_pos = fig.add_subplot(gs[base_idx, 0])
            first_ax = ax_pos
        else:
            ax_pos = fig.add_subplot(gs[base_idx, 0], sharex=first_ax)
        ax_neg = fig.add_subplot(gs[base_idx + 1, 0], sharex=first_ax)
        ax_ctrl = fig.add_subplot(gs[base_idx + 2, 0], sharex=first_ax)

        pos_color = RUN_COLORS[f"{run_name}_pos"]
        neg_color = RUN_COLORS[f"{run_name}_neg"]

        ax_pos.plot(df["curr_time"], df["ref_pos"], color="black", label="ref_pos")
        ax_pos.plot(df["curr_time"], df["sen_pos"], color=pos_color, label=f"{run_name}_pos")
        ax_pos.grid(True)
        ax_pos.legend(loc="upper right")
        ax_pos.set_ylabel(f"{run_name} Pos [kPa]")

        ax_neg.plot(df["curr_time"], df["ref_neg"], color="black", label="ref_neg")
        ax_neg.plot(df["curr_time"], df["sen_neg"], color=neg_color, label=f"{run_name}_neg")
        ax_neg.grid(True)
        ax_neg.legend(loc="upper right")
        ax_neg.set_ylabel(f"{run_name} Neg [kPa]")

        ax_ctrl.plot(df["curr_time"], df["ctrl_pos"], color=pos_color, label=f"{run_name}_ctrl_pos")
        ax_ctrl.plot(df["curr_time"], df["ctrl_neg"], color=neg_color, label=f"{run_name}_ctrl_neg")
        ax_ctrl.grid(True)
        ax_ctrl.legend(loc="upper right")
        ax_ctrl.set_ylabel(f"{run_name} Control")
        if idx == len(run_dfs) - 1:
            ax_ctrl.set_xlabel("Time [sec]")

    title_metrics = " | ".join(
        f"{name} pos={m['rmse_pos']:.3f}, neg={m['rmse_neg']:.3f}"
        for name, m in metrics.items()
    )
    fig.suptitle(f"{save_name} | {title_metrics}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{save_name}.png", dpi=150)

    print(f"[INFO] Saved baseline: {out_dir}")
    for run_name, run_metrics in metrics.items():
        print(f"[INFO] {run_name} metrics: {run_metrics}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-mode", choices=["pid_rl", "pid"], default="pid_rl")
    ap.add_argument(
        "--runs",
        nargs="+",
        choices=["pid_sim2", "pid_sim", "rl_sim2"],
        default=None,
    )
    ap.add_argument("--model-name", default="0427_lib2_Ours_2")
    ap.add_argument("--ref", choices=["random", "sin", "cos", "step"], default="random")
    ap.add_argument("--max-time", type=float, default=100.0)
    ap.add_argument("--freq", type=float, default=50.0)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--scale", action="store_true", default=True)
    ap.add_argument("--no-scale", dest="scale", action="store_false")
    ap.add_argument("--save-name", default=None)
    args = ap.parse_args()

    run_dfs = run_selected(args)
    save_outputs(run_dfs, args)


if __name__ == "__main__":
    main()
