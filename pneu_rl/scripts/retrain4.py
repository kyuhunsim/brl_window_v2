import os

from pneu_ref.random_ref import RandomRef
from pneu_ref.disp_ref import PressureDiffDisplacementRef, PressureDisplacementRef, RandomDispRef
from pneu_env.env4 import PneuEnv4
from pneu_env.sim4 import PneuSim
from pneu_env.pred4 import PneuPred
from pneu_env.real.real4 import PneuReal
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import (
    delete_lines,
    color,
    get_pkg_path,
    save_yaml,
    load_yaml,
)


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

DISP_REWARD_DEFAULTS = dict(
    disp_prev_rwd_coeff=0.0,
    disp_curr_rwd_coeff=0.2,
    disp_fut_rwd_coeff=0.0,
    disp_pred_rwd_coeff=0.0,
    disp_vel_rwd_coeff=0.0,
)

DISP_REWARD_WITH_PRED = dict(
    disp_prev_rwd_coeff=0.0,
    disp_curr_rwd_coeff=0.2,
    disp_fut_rwd_coeff=0.005,
    disp_pred_rwd_coeff=0.08,
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


print(color("[INPUT] Retrain Model:", "blue"))
models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
for i, model in enumerate(models):
    print(color(f"\t{i+1}. {model}", "yellow"))
print(color("\t---", "blue"))
model_idx = int(input(color("\tMODEL: ", "blue"))) - 1
model_name = models[model_idx]
delete_lines(len(models) + 3)
print(f"[ INFO] Model name: {model_name}")

retrain_model_name = input(color("[INPUT] Retrain Model Name: ", "blue"))
delete_lines(1)
if retrain_model_name == "":
    print("[ INFO] Retrain model name is set automatically.")
else:
    print(f"[ INFO] Retrain model name: {retrain_model_name}")

print(color("[INPUT] Observation Mode:", "blue"))
print(color("\t1. Sim", "yellow"))
print(color("\t2. Real", "yellow"))
print(color("\t---", "blue"))
obs_mode = input(color("\tOBS: ", "blue"))
delete_lines(5)

kwargs = load_yaml(model_name)

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
kwargs.setdefault("disp_ref", DEFAULT_DISP_REF.copy())
kwargs.setdefault("ref_mode", "pressure_diff_displacement")
kwargs.setdefault("pressure_diff_disp_ref", DEFAULT_PRESSURE_DIFF_DISP_REF.copy())
kwargs["obs"].setdefault("init_length", TUNED_INIT_CELL_LENGTH_M)
kwargs["obs"].setdefault("init_pos_press", TUNED_INIT_POS_PRESS_KPA)
kwargs["obs"].setdefault("init_neg_press", TUNED_INIT_NEG_PRESS_KPA)
kwargs["obs"].setdefault("init_act_pos_press", TUNED_INIT_ACT_POS_PRESS_KPA)
kwargs["obs"].setdefault("init_act_neg_press", TUNED_INIT_ACT_NEG_PRESS_KPA)
if kwargs.get("pred") is not None:
    kwargs["pred"].setdefault("init_length", TUNED_INIT_CELL_LENGTH_M)
    kwargs["pred"].setdefault("init_pos_press", TUNED_INIT_POS_PRESS_KPA)
    kwargs["pred"].setdefault("init_neg_press", TUNED_INIT_NEG_PRESS_KPA)
    kwargs["pred"].setdefault("init_act_pos_press", TUNED_INIT_ACT_POS_PRESS_KPA)
    kwargs["pred"].setdefault("init_act_neg_press", TUNED_INIT_ACT_NEG_PRESS_KPA)

kwargs["parent_model"] = model_name
kwargs["retrain_epi"] = 200
kwargs["type"] = "simulation" if obs_mode == "1" else "real"

if obs_mode == "2":
    kwargs["env"]["verbose"] = False
    kwargs["model"]["update_every"] = 10
    print(color("[INPUT] PID on for this real retrain?", "blue"))
    print(color("\t1. Yes", "yellow"))
    print(color("\t2. No", "yellow"))
    print(color("\t---", "blue"))
    pid_mode = input(color("\tPID: ", "blue"))
    delete_lines(5)
    if pid_mode == "1":
        kwargs["pid"] = REAL_RETRAIN_PID.copy()
        print("[ INFO] PID: enabled for this real retrain")
    elif pid_mode == "2":
        kwargs["pid"] = None
        print("[ INFO] PID: disabled for this real retrain")
    else:
        raise ValueError(color(f"[ERROR] Unknown PID mode: {pid_mode}", "red"))

    rest_angle_input = input(
        color("[INPUT] Max extension encoder angle deg (blank: tuned 8.073mm): ", "blue")
    ).strip()
    delete_lines(1)
    if rest_angle_input:
        kwargs["obs"]["encoder_rest_angle"] = float(rest_angle_input)
        print(f"[ INFO] Encoder rest angle: {kwargs['obs']['encoder_rest_angle']:.6f} deg")
    else:
        print("[ INFO] Encoder rest angle: not set; using tuned initial displacement")

# Keep the selected model's reference distribution for sim2real.
# For 0528Ours this matches the no-PID simulation training condition.

kwargs["temporal_weight_hardening"] = dict(
    initial_weight=1.25,
    max_weight=1.25,
)
total_steps = 100 * kwargs["model"]["horizon"]
kwargs["temporal_weight_hardening"]["rate"] = (
    kwargs["temporal_weight_hardening"]["max_weight"]
    - kwargs["temporal_weight_hardening"]["initial_weight"]
) / total_steps

if obs_mode == "1":
    print("[ INFO] Observation Mode: Simulation")
    obs = PneuSim(**kwargs["obs"])
elif obs_mode == "2":
    print("[ INFO] Observation Mode: Real")
    real_kwargs = dict(kwargs["obs"])
    real_kwargs.setdefault("auto_zero_encoder", True)
    real_kwargs.setdefault("reset_encoder_zero_each_episode", False)
    real_kwargs.setdefault("initial_displacement_mm", TUNED_INITIAL_CONTRACTION_MM)
    real_kwargs.setdefault("clamp_displacement_ref", True)
    obs = PneuReal(**real_kwargs)
else:
    raise ValueError(color(f"[ERROR] Unknown observation mode: {obs_mode}", "red"))

pred = PneuPred(**kwargs["pred"]) if kwargs["pred"] is not None else None
pressure_ref = RandomRef(**kwargs["rnd_ref"])
if kwargs.get("ref_mode") == "pressure_diff_displacement":
    ref = PressureDiffDisplacementRef(
        pressure_ref,
        **kwargs["pressure_diff_disp_ref"],
    )
else:
    displacement_ref = RandomDispRef(**kwargs["disp_ref"])
    ref = PressureDisplacementRef(pressure_ref, displacement_ref)
env = PneuEnv4(
    obs=obs,
    ref=ref,
    pred=pred,
    **kwargs["env"],
)

if kwargs.get("pid") is not None:
    env.set_pid(**kwargs["pid"])
else:
    print("[ INFO] PID: disabled")

model = SAC(env, **kwargs["model"])
load_existing_buffer = obs_mode != "2"
model.load(name=model_name, train=True, load_buffer=load_existing_buffer)

retrain_model_name = model.set_retrain(
    retrain_model_name,
    load_buffer=load_existing_buffer,
    copy_buffer=load_existing_buffer,
)
if "alpha" in kwargs:
    model.set_alpha(**kwargs["alpha"])
if "temporal_weight_hardening" in kwargs:
    model.set_temporal_weight_hardening(**kwargs["temporal_weight_hardening"])
model.clear_buffer()

save_yaml(retrain_model_name, kwargs, "retrain_cfg.yaml")

try:
    model.train(episode=kwargs["retrain_epi"])
finally:
    env.close()
