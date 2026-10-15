from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pneu_env.env5 import PneuEnv5
from pneu_env.pred5 import PneuPred
from pneu_env.real.real5 import PneuReal
from pneu_env.sim5 import PneuSim
from pneu_ref.disp_ref import PressureDiffDisplacementRef
from pneu_ref.random_ref import RandomRef
from pneu_ref.ref5 import FlowAwarePressureDisplacementRef
from pneu_rl.sac_with_loss5 import SAC
from pneu_utils.utils import get_pkg_path

from train5 import apply_tuned_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain controller5 in sim or real")
    parser.add_argument("model_name")
    parser.add_argument("--obs", choices=("sim", "real"), default="sim")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--rest-angle", type=float, default=None)
    return parser.parse_args()


def load_config(model_name: str) -> dict:
    path = Path(get_pkg_path("pneu_rl")) / "models" / model_name / "cfg.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, args: argparse.Namespace) -> PneuEnv5:
    obs_kwargs = dict(cfg["obs"], controller=cfg["controller"])
    if args.obs == "real":
        if args.rest_angle is not None:
            obs_kwargs["encoder_rest_angle"] = args.rest_angle
        obs_kwargs.setdefault("auto_zero_encoder", True)
        obs_kwargs.setdefault("reset_encoder_zero_each_episode", False)
        obs = PneuReal(**obs_kwargs)
    else:
        obs = PneuSim(**obs_kwargs)
        apply_tuned_params(obs)

    pred = None
    if cfg.get("pred") is not None:
        pred_kwargs = dict(cfg["pred"])
        for key in (
            "init_length", "init_pos_press", "init_neg_press",
            "init_act_pos_press", "init_act_neg_press",
        ):
            pred_kwargs.setdefault(key, cfg["obs"][key])
        pred_kwargs["controller"] = cfg["controller"]
        pred = PneuPred(**pred_kwargs)
        apply_tuned_params(pred)
    pressure_ref = RandomRef(**cfg["rnd_ref"])
    base_ref = PressureDiffDisplacementRef(pressure_ref, **cfg["pressure_disp_ref"])
    ref = FlowAwarePressureDisplacementRef(base_ref, **cfg["ref5"])
    return PneuEnv5(obs=obs, pred=pred, ref=ref, **cfg["env"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_name)
    cfg["model"]["horizon"] = args.horizon or cfg["model"]["horizon"]
    env = build_env(cfg, args)
    model = SAC(env, **cfg["model"])
    model.load(name=args.model_name, train=True, load_buffer=args.obs == "sim")
    try:
        model.train(episode=args.episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()
