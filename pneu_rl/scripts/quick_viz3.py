import argparse
import os
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.env3 import PneuEnv3
from pneu_env.pred3 import PneuPred
from pneu_env.sim3 import PneuSim
from pneu_ref.random_ref import RandomRef
from pneu_rl.sac import SAC
from pneu_utils.utils import get_pkg_path, load_yaml


def info_rows(infos: deque) -> list[dict]:
    rows = []
    for info in infos:
        obs = info["obs"]
        rows.append(
            dict(
                curr_time=obs["curr_time"],
                press_pos=obs["sen_pos"],
                press_neg=obs["sen_neg"],
                act_pos_press=obs["sen_act_pos"],
                act_neg_press=obs["sen_act_neg"],
                ref_act_pos=obs["ref_act_pos"],
                ref_act_neg=obs["ref_act_neg"],
                ctrl1=obs["ctrl_pos"],
                ctrl2=obs["ctrl_neg"],
                ctrl3=obs["ctrl_act_pos_in"],
                ctrl4=obs["ctrl_act_pos_out"],
                ctrl5=obs["ctrl_act_neg_in"],
                ctrl6=obs["ctrl_act_neg_out"],
            )
        )
    return rows


def save_and_plot(rows: list[dict], save_name: str, cfg: dict) -> None:
    out_dir = f'{get_pkg_path("pneu_rl")}/exp/{save_name}'
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/{save_name}.csv", index=False)
    with open(f"{out_dir}/cfg.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    t = df["curr_time"].to_numpy()
    axes[0].plot(t, df["press_pos"], label="press_pos")
    axes[0].plot(t, df["press_neg"], label="press_neg")
    axes[0].set_ylabel("Chamber [kPa]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, df["ref_act_pos"], color="black", label="ref_act_pos")
    axes[1].plot(t, df["act_pos_press"], color="red", label="act_pos_press")
    axes[1].set_ylabel("Act Pos [kPa]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, df["ref_act_neg"], color="black", label="ref_act_neg")
    axes[2].plot(t, df["act_neg_press"], color="blue", label="act_neg_press")
    axes[2].set_ylabel("Act Neg [kPa]")
    axes[2].grid(True)
    axes[2].legend()

    for i in range(1, 7):
        axes[3].plot(t, df[f"ctrl{i}"], label=f"ctrl{i}")
    axes[3].set_xlabel("Time [sec]")
    axes[3].set_ylabel("Control")
    axes[3].grid(True)
    axes[3].legend(ncol=3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{save_name}.png", dpi=150)
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--save-name", default=None)
    args = ap.parse_args()

    kwargs = load_yaml(args.model_name)
    obs = PneuSim(**kwargs["obs"])
    pred = PneuPred(**kwargs["pred"]) if kwargs["pred"] is not None else None
    ref = RandomRef(**kwargs["rnd_ref"])
    env = PneuEnv3(
        obs=obs,
        pred=pred,
        ref=ref,
        **kwargs["env"],
    )

    model = SAC(env=env, **kwargs["model"])
    model.set_logger(args.model_name)
    model.load_model(path=model.logger.model_path, evaluate=True)

    state, info = env.reset()
    infos = deque([info])
    for _ in range(args.steps):
        action = model.predict(state)
        state, _, _, _, info = env.step(action)
        infos.append(info)

    env.close()

    save_name = args.save_name
    if not save_name:
        stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        save_name = f"{stamp}_{args.model_name}_Simulation"
    save_and_plot(info_rows(infos), save_name, kwargs)


if __name__ == "__main__":
    main()
