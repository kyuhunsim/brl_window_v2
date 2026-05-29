import os

from pneu_ref.random_ref import RandomRef
from pneu_env.env3 import PneuEnv3
from pneu_env.sim3 import PneuSim
from pneu_env.pred3 import PneuPred
from pneu_env.real.real9 import PneuReal
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import (
    delete_lines,
    color,
    get_pkg_path,
    save_yaml,
    load_yaml,
)


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

kwargs["parent_model"] = model_name
kwargs["retrain_epi"] = 50
kwargs["type"] = "simulation" if obs_mode == "1" else "real"

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
    obs = PneuReal(**kwargs["obs"])
else:
    raise ValueError(color(f"[ERROR] Unknown observation mode: {obs_mode}", "red"))

pred = PneuPred(**kwargs["pred"]) if kwargs["pred"] is not None else None
ref = RandomRef(**kwargs["rnd_ref"])
env = PneuEnv3(
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
model.load(name=model_name, train=True)

retrain_model_name = model.set_retrain(retrain_model_name)
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
