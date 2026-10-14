from __future__ import annotations

import argparse

import numpy as np

from train5 import build_env, default_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short PID-only controller5 baseline")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--no-pred", action="store_true")
    parser.add_argument("--residual-mode", default="reserve", choices=("reserve", "valve"))
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    cfg["env"]["verbose"] = False
    env = build_env(cfg)
    _, _ = env.reset()
    rows = []
    try:
        for _ in range(args.steps):
            _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            rows.append((reward, info["reward"]["disp_error"], info["reward"]["pos_error"], info["reward"]["neg_error"]))
    finally:
        env.close()
    data = np.asarray(rows, dtype=np.float64)
    rmse = np.sqrt(np.mean(data[:, 1:] ** 2, axis=0))
    print(f"steps={len(data)} reward_mean={np.mean(data[:,0]):.6f}")
    print(f"rmse disp={rmse[0]:.6f} act_pos={rmse[1]:.6f} act_neg={rmse[2]:.6f}")


if __name__ == "__main__":
    main()
