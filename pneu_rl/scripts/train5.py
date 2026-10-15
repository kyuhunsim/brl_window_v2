from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pneu_env.env5 import PneuEnv5
from pneu_env.pred5 import PneuPred
from pneu_env.sim5 import PneuSim
from pneu_ref.disp_ref import PressureDiffDisplacementRef
from pneu_ref.random_ref import RandomRef
from pneu_ref.ref5 import FlowAwarePressureDisplacementRef
from pneu_rl.sac_with_loss5 import SAC
from pneu_utils.utils import get_pkg_path, save_yaml


TUNED_DIR = (
    Path(get_pkg_path("pneu_env"))
    / "coeff_result/260624_11_21_35_tuner4_output48plus49_linit34807_leak_tuned"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flow-aware residual controller 5")
    parser.add_argument("model_name")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--residual-mode", choices=("reserve", "valve"), default="reserve")
    parser.add_argument("--no-pred", action="store_true")
    return parser.parse_args()


def default_config(args: argparse.Namespace) -> dict:
    return dict(
        obs=dict(
            freq=50, delay=0.1, noise=False, scale=True, scale_low=0.8, scale_high=1.0,
            init_length=0.03192691119048332/2.0,
            init_pos_press=135.98092651367188, init_neg_press=42.16064834594727,
            init_act_pos_press=110.0009307861328, init_act_neg_press=87.99779510498047,
        ),
        pred=None if args.no_pred else dict(
            freq=50, delay=0.0, noise=False, scale=True, scale_low=0.8, scale_high=1.0
        ),
        rnd_ref=dict(
            pos_min_off=112, pos_max_off=120, neg_min_off=80, neg_max_off=88,
            pos_max_ts=5, neg_max_ts=5, pos_max_amp=4, neg_max_amp=2,
            pos_max_per=10, neg_max_per=10,
        ),
        pressure_disp_ref=dict(
            dp_to_disp_slope=0.6537588686709214,
            dp_to_disp_intercept=-1.7580850354058029,
            min_disp=None, max_disp=None,
        ),
        ref5=dict(
            base_margin_pos=25.0, base_margin_neg=35.0,
            rate_margin_pos=0.10, rate_margin_neg=0.10,
            actuator_volume_pos_m3=5.0e-5, actuator_volume_neg_m3=5.0e-5,
            valve_flow_coeff=5.0e-7, flow_safety_factor=1.25,
            ch_pos_min=110.0, ch_pos_max=500.0, ch_neg_min=5.0, ch_neg_max=90.0,
        ),
        controller=dict(
            chamber_pos=dict(kp=0.04, ki=0.0, kd=0.0, deadband=1.0),
            chamber_neg=dict(kp=0.04, ki=0.0, kd=0.0, deadband=1.0),
            actuator_pos=dict(kp=0.03, ki=0.0, kd=0.0, deadband=0.5),
            actuator_neg=dict(kp=0.03, ki=0.0, kd=0.0, deadband=0.5),
            min_open=0.0, max_open=1.0, valve_gamma=1.0,
            reserve_freeze_threshold=2.0,
        ),
        env=dict(
            num_prev=10, num_pred=15, num_act=5, verbose=False,
            episode_carry_state=True, include_displacement_velocity_obs=False,
            include_prev_action_obs=True, include_pressure_diff_velocity_obs=True,
            residual_mode=args.residual_mode,
            rwd_kwargs=dict(
                pos_curr_rwd_coeff=0.3, neg_curr_rwd_coeff=0.3,
                disp_curr_rwd_coeff=0.2, diff_curr_rwd_coeff=0.1,
                ch_pos_shortage_rwd_coeff=0.01, ch_neg_shortage_rwd_coeff=0.01,
                conflict_rwd_coeff=0.5, action_delta_rwd_coeff=0.05,
                residual_rwd_coeff=0.01, pred_rwd_coeff=0.05,
            ),
        ),
        model=dict(
            learning_rate=3e-4, gamma=0.9, tau=0.005, alpha=0.5,
            automatic_entropy_tunning=True, hidden_dim=256, buffer_size=500000,
            batch_size=128, epoch=1, horizon=args.horizon, start_epi=10,
            max_grad_norm=0.5, log_std_min=-10, log_std_max=1,
            temporal_weight=2.0, spatial_weight=0.6, noise_std=0.0,
            train_diag_interval=1, train_diag_atm_band=8.0,
        ),
        epi=args.episodes,
    )


def apply_tuned_params(sim: PneuSim) -> None:
    params_path = TUNED_DIR / "params.yaml"
    if not params_path.exists():
        return
    with params_path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    sim.set_discharge_coeff(float(params["pump_in_coeff"]), float(params["pump_out_coeff"]))
    sim.set_leak_coefficients(
        pos_atm=float(params["leak_pos_atm"]),
        neg_atm=float(params["leak_neg_atm"]),
        cross=float(params["leak_cross"]),
    )


def build_env(cfg: dict) -> PneuEnv5:
    obs_kwargs = dict(cfg["obs"], controller=cfg["controller"])
    obs = PneuSim(**obs_kwargs)
    pred = None
    if cfg["pred"] is not None:
        pred_kwargs = dict(cfg["pred"])
        for key in (
            "init_length", "init_pos_press", "init_neg_press",
            "init_act_pos_press", "init_act_neg_press",
        ):
            pred_kwargs.setdefault(key, cfg["obs"][key])
        pred = PneuPred(**dict(pred_kwargs, controller=cfg["controller"]))
    apply_tuned_params(obs)
    if pred is not None:
        apply_tuned_params(pred)
    pressure_ref = RandomRef(**cfg["rnd_ref"])
    base_ref = PressureDiffDisplacementRef(pressure_ref, **cfg["pressure_disp_ref"])
    ref = FlowAwarePressureDisplacementRef(base_ref, **cfg["ref5"])
    return PneuEnv5(obs=obs, pred=pred, ref=ref, **cfg["env"])


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    env = build_env(cfg)
    model = SAC(env=env, **cfg["model"])
    model.set_logger(args.model_name)
    save_yaml(args.model_name, cfg)
    try:
        model.train(episode=cfg["epi"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
