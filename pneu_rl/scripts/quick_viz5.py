from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pneu_rl.sac_with_loss5 import SAC
from pneu_utils.utils import get_pkg_path
from pneu_rl.exp_paths import ensure_model_exp_dir

from retrain5 import build_env, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and plot controller5")
    parser.add_argument("model_name")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.obs = "sim"
    args.rest_angle = None
    args.horizon = args.steps
    cfg = load_config(args.model_name)
    env = build_env(cfg, args)
    model = SAC(env, **cfg["model"])
    model.load(name=args.model_name, train=False, load_buffer=False)
    state, _ = env.reset()
    rows = []
    try:
        for _ in range(args.steps):
            action = model.evaluate_action(state)
            state, reward, _, _, info = env.step(action)
            obs = info["obs"]
            rows.append(dict(
                time=obs["curr_time"], reward=reward,
                ch_pos=obs["sen_pos"], ch_neg=obs["sen_neg"],
                act_pos=obs["sen_act_pos"], act_neg=obs["sen_act_neg"],
                ref_ch_pos=obs["ref_ch_pos"], ref_ch_neg=obs["ref_ch_neg"],
                ref_act_pos=obs["ref_act_pos"], ref_act_neg=obs["ref_act_neg"],
                displacement=obs.get("sen_env_displacement", 0.0),
                ref_displacement=obs["ref_displacement"],
                residual_pos=obs["residual_ctrl"][0], residual_neg=obs["residual_ctrl"][1],
            ))
    finally:
        env.close()
    df = pd.DataFrame(rows)
    out = Path(args.output) if args.output else ensure_model_exp_dir(
        args.model_name, f"quick_viz5_{args.steps}step"
    )
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"rollout.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(df.time, df.ch_pos, label="ch pos"); axes[0].plot(df.time, df.ref_ch_pos, "--", label="ref")
    axes[0].plot(df.time, df.ch_neg, label="ch neg"); axes[0].plot(df.time, df.ref_ch_neg, "--", label="ref")
    axes[1].plot(df.time, df.act_pos, label="act pos"); axes[1].plot(df.time, df.ref_act_pos, "--", label="ref")
    axes[1].plot(df.time, df.act_neg, label="act neg"); axes[1].plot(df.time, df.ref_act_neg, "--", label="ref")
    axes[2].plot(df.time, df.displacement, label="disp"); axes[2].plot(df.time, df.ref_displacement, "--", label="ref")
    for axis in axes: axis.grid(True); axis.legend()
    fig.tight_layout(); fig.savefig(out/"rollout.png", dpi=150); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
