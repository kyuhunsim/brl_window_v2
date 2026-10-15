from __future__ import annotations

import argparse
import time

import numpy as np

from retrain5 import build_env, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Controller5 loop-rate smoke test")
    parser.add_argument("model_name")
    parser.add_argument("--obs", choices=("sim", "real"), default="sim")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rest-angle", type=float, default=None)
    args = parser.parse_args()
    args.horizon = args.steps
    args.episodes = 1
    env = build_env(load_config(args.model_name), args)
    env.reset()
    samples = []
    try:
        for _ in range(args.steps):
            start = time.perf_counter()
            env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            samples.append(time.perf_counter()-start)
    finally:
        env.close()
    samples = np.asarray(samples)
    print(
        f"steps={len(samples)} mean_ms={samples.mean()*1000:.3f} "
        f"p95_ms={np.percentile(samples,95)*1000:.3f} hz={1/samples.mean():.2f}"
    )


if __name__ == "__main__":
    main()

