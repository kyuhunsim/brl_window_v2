import argparse
import time
from pathlib import Path

import numpy as np

from pneu_ref.random_ref import RandomRef
from pneu_ref.disp_ref import PressureDiffDisplacementRef, PressureDisplacementRef, RandomDispRef
from pneu_env.env4 import PneuEnv4
from pneu_env.sim4 import PneuSim
from pneu_env.pred4 import PneuPred
from pneu_env.real.real4 import PneuReal
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import get_pkg_path, load_yaml

# python3 pneu_rl/scripts/retrain4_hz_test.py \
#   --obs real \
#   --model 0626_lib4_Ours_no_vel_action_penalty \
#   --episodes 10 \
#   --horizon 512 \
#   --update-every 5 \
#   --pid off \
#   --rest-angle 311.63

DEFAULT_MODEL = "0626_lib4_Ours_no_vel_action_penalty"

DEFAULT_DISP_REF = dict(
    max_ts=5,
    max_amp=3.0,
    max_per=10.0,
    min_off=10.0,
    max_off=18.0,
    seed=61099,
)

DEFAULT_PRESSURE_DIFF_DISP_REF = dict(
    dp_to_disp_slope=0.6537588686709214,
    dp_to_disp_intercept=-1.7580850354058029,
    min_disp=None,
    max_disp=None,
)

DISP_REWARD_WITH_PRED = dict(
    disp_prev_rwd_coeff=0.0,
    disp_curr_rwd_coeff=0.2,
    disp_fut_rwd_coeff=0.005,
    disp_pred_rwd_coeff=0.08,
    disp_vel_rwd_coeff=0.0,
)

DISP_REWARD_DEFAULTS = dict(
    disp_prev_rwd_coeff=0.0,
    disp_curr_rwd_coeff=0.2,
    disp_fut_rwd_coeff=0.0,
    disp_pred_rwd_coeff=0.0,
    disp_vel_rwd_coeff=0.0,
)

REAL_RETRAIN_PID = dict(
    Kp_act_pos_in=0.0,
    Ki_act_pos_in=0.002,
    Kd_act_pos_in=0.0,
    Kp_act_pos_out=0.0,
    Ki_act_pos_out=0.002,
    Kd_act_pos_out=0.0,
    Kp_act_neg_in=0.0,
    Ki_act_neg_in=0.002,
    Kd_act_neg_in=0.0,
    Kp_act_neg_out=0.0,
    Ki_act_neg_out=0.002,
    Kd_act_neg_out=0.0,
    Ka=1,
)

TUNED_L_INIT_TOTAL_M = 0.03192691119048332
TUNED_INIT_CELL_LENGTH_M = TUNED_L_INIT_TOTAL_M / 2.0
TUNED_INITIAL_CONTRACTION_MM = (0.04 - TUNED_L_INIT_TOTAL_M) * 1000.0
TUNED_INIT_POS_PRESS_KPA = 135.98092651367188
TUNED_INIT_NEG_PRESS_KPA = 42.16064834594727
TUNED_INIT_ACT_POS_PRESS_KPA = 110.0009307861328
TUNED_INIT_ACT_NEG_PRESS_KPA = 87.99779510498047


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short lib4 retrain throughput smoke test. Does not save model weights."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--obs", choices=("real", "sim"), default="real")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--update-every", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--pid", choices=("on", "off"), default="off")
    parser.add_argument("--rest-angle", type=float, default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--deterministic-action", action="store_true")
    return parser.parse_args()


def apply_retrain_defaults(kwargs: dict, args: argparse.Namespace) -> dict:
    kwargs["env"]["rwd_kwargs"].update(
        action_delta_rwd_coeff=0.08,
        conflict_rwd_coeff=0.15,
    )
    disp_reward = DISP_REWARD_WITH_PRED if kwargs.get("pred") is not None else DISP_REWARD_DEFAULTS
    kwargs["env"]["rwd_kwargs"].update(disp_reward)
    kwargs["env"].setdefault("displacement_obs_key", "sen_total_contraction_from_rest")
    kwargs["env"].setdefault("displacement_scale", 1000.0)
    kwargs["env"].setdefault("velocity_scale", 1000.0)
    kwargs["env"].setdefault("fixed_chamber_ctrl", None)
    kwargs["env"].setdefault("steady_chamber_ctrl", None)
    kwargs["env"]["verbose"] = False

    kwargs.setdefault("disp_ref", DEFAULT_DISP_REF.copy())
    kwargs.setdefault("ref_mode", "pressure_diff_displacement")
    kwargs.setdefault("pressure_diff_disp_ref", DEFAULT_PRESSURE_DIFF_DISP_REF.copy())

    kwargs["obs"].setdefault("init_length", TUNED_INIT_CELL_LENGTH_M)
    kwargs["obs"].setdefault("init_pos_press", TUNED_INIT_POS_PRESS_KPA)
    kwargs["obs"].setdefault("init_neg_press", TUNED_INIT_NEG_PRESS_KPA)
    kwargs["obs"].setdefault("init_act_pos_press", TUNED_INIT_ACT_POS_PRESS_KPA)
    kwargs["obs"].setdefault("init_act_neg_press", TUNED_INIT_ACT_NEG_PRESS_KPA)
    if args.rest_angle is not None:
        kwargs["obs"]["encoder_rest_angle"] = args.rest_angle

    if kwargs.get("pred") is not None:
        kwargs["pred"].setdefault("init_length", TUNED_INIT_CELL_LENGTH_M)
        kwargs["pred"].setdefault("init_pos_press", TUNED_INIT_POS_PRESS_KPA)
        kwargs["pred"].setdefault("init_neg_press", TUNED_INIT_NEG_PRESS_KPA)
        kwargs["pred"].setdefault("init_act_pos_press", TUNED_INIT_ACT_POS_PRESS_KPA)
        kwargs["pred"].setdefault("init_act_neg_press", TUNED_INIT_ACT_NEG_PRESS_KPA)

    kwargs["model"]["horizon"] = args.horizon
    kwargs["model"]["update_every"] = args.update_every
    kwargs["model"]["epoch"] = args.epoch
    return kwargs


def build_env(kwargs: dict, args: argparse.Namespace) -> PneuEnv4:
    if args.obs == "real":
        if not args.no_wait:
            input(
                "[ACTION] Set rest angle in cfg/--rest-angle, turn pump on, "
                "wait until initial pressures/angle are stable, then press Enter: "
            )
        real_kwargs = dict(kwargs["obs"])
        real_kwargs.setdefault("auto_zero_encoder", True)
        real_kwargs.setdefault("reset_encoder_zero_each_episode", False)
        real_kwargs.setdefault("initial_displacement_mm", TUNED_INITIAL_CONTRACTION_MM)
        real_kwargs.setdefault("clamp_displacement_ref", True)
        obs = PneuReal(**real_kwargs)
    else:
        obs = PneuSim(**kwargs["obs"])

    pred = PneuPred(**kwargs["pred"]) if kwargs.get("pred") is not None else None
    pressure_ref = RandomRef(**kwargs["rnd_ref"])
    if kwargs.get("ref_mode") == "pressure_diff_displacement":
        ref = PressureDiffDisplacementRef(
            pressure_ref,
            **kwargs["pressure_diff_disp_ref"],
        )
    else:
        displacement_ref = RandomDispRef(**kwargs["disp_ref"])
        ref = PressureDisplacementRef(pressure_ref, displacement_ref)

    env = PneuEnv4(obs=obs, ref=ref, pred=pred, **kwargs["env"])
    if args.pid == "on":
        env.set_pid(**REAL_RETRAIN_PID)
    return env


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    model_dir = Path(get_pkg_path("pneu_rl")) / "models" / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"model folder not found: {model_dir}")

    kwargs = apply_retrain_defaults(load_yaml(args.model), args)
    env = build_env(kwargs, args)
    model = SAC(env, **kwargs["model"])
    model.load(name=args.model, train=True, load_buffer=False)
    model.clear_buffer()

    steps_target = args.episodes * args.horizon
    warmup_steps = max(int(args.warmup_steps), 0)
    total_target = warmup_steps + steps_target
    local_steps = 0
    update_count = 0
    reward_sum = 0.0

    action_times: list[float] = []
    env_times: list[float] = []
    update_times: list[float] = []
    loop_times: list[float] = []

    try:
        state, _ = env.reset()
        t_start = time.perf_counter()
        while local_steps < total_target:
            if local_steps == warmup_steps and warmup_steps > 0:
                print(f"[ INFO] Warmup done at {warmup_steps} steps. Updates start now.")

            t0 = time.perf_counter()
            if args.deterministic_action:
                action = model.evaluate_action(state)
            else:
                action = model.select_action(state)
            t1 = time.perf_counter()

            next_state, reward, done, _, _ = env.step(action)
            t2 = time.perf_counter()

            model.buffer.add(state, action, reward, next_state, done)
            state = next_state
            reward_sum += float(reward)

            do_update = (
                not args.no_update
                and local_steps >= warmup_steps
                and len(model.buffer) > model.batch_size
                and local_steps % model.update_every == 0
            )
            if do_update:
                for _ in range(model.epoch):
                    model.update_parameters()
                    update_count += 1
            t3 = time.perf_counter()

            action_times.append(t1 - t0)
            env_times.append(t2 - t1)
            update_times.append(t3 - t2)
            loop_times.append(t3 - t0)

            local_steps += 1
            if done or (local_steps < total_target and local_steps % args.horizon == 0):
                state, _ = env.reset()

        elapsed = time.perf_counter() - t_start
    finally:
        env.close()

    train_steps = max(total_target - warmup_steps, 0)
    print("[ RESULT] retrain4_hz_test")
    print(f"  model: {args.model}")
    print(f"  obs: {args.obs}, pid: {args.pid}")
    print(f"  warmup_steps: {warmup_steps}, train_steps: {train_steps}")
    print(f"  update_every: {model.update_every}, epoch: {model.epoch}, updates: {update_count}")
    print(f"  elapsed_sec: {elapsed:.3f}")
    print(f"  total_hz: {total_target / elapsed:.2f}")
    print(f"  reward_sum: {reward_sum:.3f}")
    print(f"  buffer_size: {len(model.buffer)}")
    print(
        "  loop_ms mean/p50/p90: "
        f"{np.mean(loop_times) * 1e3:.3f} / {pct(loop_times, 50) * 1e3:.3f} / {pct(loop_times, 90) * 1e3:.3f}"
    )
    print(
        "  action_ms mean/p90: "
        f"{np.mean(action_times) * 1e3:.3f} / {pct(action_times, 90) * 1e3:.3f}"
    )
    print(
        "  env_step_ms mean/p90: "
        f"{np.mean(env_times) * 1e3:.3f} / {pct(env_times, 90) * 1e3:.3f}"
    )
    print(
        "  update_ms mean/p90: "
        f"{np.mean(update_times) * 1e3:.3f} / {pct(update_times, 90) * 1e3:.3f}"
    )


if __name__ == "__main__":
    main()
