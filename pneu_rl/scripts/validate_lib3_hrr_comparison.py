"""Local numerical preflight for matched HRR / HRR+DR training.

Checks physical invariants and runnable SAC updates, not global stability,
reference reachability, or the expected quality of a future trained policy.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import time

from train_lib3_hrr_comparison import (
    ROOT, SOURCE, EXPERIMENT, DR_PROFILE, build_config, make_training_components,
    policy_hash, sha256, make_components, seed_everything, SAC, np, torch, yaml,
)
from check_lib3_closed_loop_feasibility import NativeStateSampler, analyze_arrays
from evaluate_lib3_screening import reset_to_reference_member
from scipy.signal import welch


IMPLEMENTATIONS = (
    "pneu_rl/scripts/train_lib3_hrr_comparison.py",
    "pneu_rl/scripts/build_lib3_randomized_plant.py",
    "pneu_rl/scripts/validate_lib3_hrr_comparison.py",
    "pneu_rl/scripts/check_lib3_closed_loop_feasibility.py",
    "pneu_rl/scripts/train_screening.py",
    "pneu_env/src/pneu_env/env3_randomized.py",
    "pneu_env/src/pneu_env/env3_all_open.py",
    "pneu_env/src/pneu_env/env3.py",
    "pneu_env/src/pneu_env/sim3.py",
    "pneu_rl/src/pneu_rl/sac_with_loss.py",
    "pneu_rl/config/domain_randomization/lib3_type1_v34_v1.yaml",
)
CHECKPOINT = SOURCE.parent / "model_epi_2000.pth"


def one_scenario(cfg, scenario):
    result = copy.deepcopy(cfg)
    result["training_dynamics_randomization"].update(
        scenarios=[copy.deepcopy(scenario)], first_nominal=False)
    result["training_dynamics_randomization"].pop("log_path", None)
    return result


def fixed_action_rollout(cfg, actions):
    seed_everything(cfg["train_seed"])
    env = make_training_components(cfg)
    sampler = NativeStateSampler(env.obs)
    try:
        state, _ = reset_to_reference_member(env, 0)
        values = [np.concatenate((state, sampler.sample()["native_state"]))]
        for action in actions:
            state, reward, done, truncated, info = env.step(action)
            if done or truncated:
                raise AssertionError("Unexpected termination in nominal parity check")
            values.append(np.concatenate((state, sampler.sample()["native_state"], [reward])))
        return values
    finally:
        env.close()


def update_smoke(cfg, directory):
    """One real SAC update from 160 simulated transitions, without model logging."""
    seed_everything(cfg["train_seed"])
    env = make_training_components(cfg)
    try:
        if env.observation_space.shape != (162,):
            raise AssertionError("HRR must have 162 observations")
        model = SAC(env=env, **cfg["model"])
        initial = policy_hash(model)
        state, _ = reset_to_reference_member(env, 0)
        for _ in range(160):
            action = model.predict(state, evaluate=False)
            next_state, reward, done, truncated, _ = env.step(action)
            if not np.isfinite(next_state).all() or not np.isfinite(reward):
                raise AssertionError("Nonfinite training transition")
            model.buffer.add(state, action, reward, next_state, done or truncated)
            model.total_steps += 1
            state = next_state
            if done or truncated:
                raise AssertionError("Unexpected termination in update smoke")
        model.update_parameters()
        losses = {"critic": float(model.current_critics_loss),
                  "actor": float(model.current_policy_loss),
                  "alpha": float(model.alpha.detach())}
        if not np.isfinite(list(losses.values())).all():
            raise AssertionError(f"Nonfinite SAC losses: {losses}")
        updated = policy_hash(model)
        if updated == initial:
            raise AssertionError("SAC update did not change the actor")
        for network in (model.policy, model.critic, model.critic_target):
            if not all(torch.isfinite(tensor).all() for tensor in network.state_dict().values()):
                raise AssertionError("Nonfinite learned weights")
        expected_action = model.predict(state, evaluate=True)
        with tempfile.TemporaryDirectory(prefix="checkpoint-smoke-", dir=directory) as temporary:
            checkpoint = str(Path(temporary) / "model.pth")
            model.save_model(checkpoint)
            restored = SAC(env=env, **cfg["model"])
            restored.load_model(checkpoint, evaluate=True)
            restored_action = restored.predict(state, evaluate=True)
            if policy_hash(restored) != updated or not np.array_equal(expected_action, restored_action):
                raise AssertionError("Checkpoint round-trip changed deterministic policy")
        return {"passed": True, "state_dim": 162, "replay_samples": len(model.buffer),
                "updates": 1, "initial_actor_sha256": initial,
                "updated_actor_sha256": updated, "losses": losses,
                "save_load_deterministic_action_max_abs_error": 0.0}
    finally:
        env.close()


def frozen_hr_model():
    cfg = yaml.safe_load(SOURCE.read_text())
    seed_everything(cfg["train_seed"])
    env = make_components(cfg)
    try:
        model = SAC(env=env, **cfg["model"])
        model.load_model(str(CHECKPOINT), evaluate=True)
        if env.observation_space.shape != (160,):
            raise AssertionError("Frozen HR must have 160 inputs")
        return model
    finally:
        env.close()


def scenario_rollout(cfg, model, scenario):
    seed_everything(cfg["train_seed"])
    env = make_training_components(one_scenario(cfg, scenario))
    sampler = NativeStateSampler(env.obs)
    steps = 2048
    arrays = {key: np.empty((steps,) + shape) for key, shape in {
        "native_state": (37,), "valve_debug": (6, 9), "mass_flowrate": (10,),
        "physical_command": (6,), "observed_pressure": (4,), "reference": (2,),
    }.items()}
    try:
        state, _ = reset_to_reference_member(env, 0)
        initial = {"pressure_kpa": env.curr_full_press.tolist(),
                   "reference_kpa": env._current_actuator_reference().tolist(),
                   "reference_minus_pressure_kpa": (env._current_actuator_reference()
                                                        - env.curr_full_press[2:]).tolist(),
                   "native_time_seconds": float(env.obs.lib.get_time_c())}
        for step in range(steps):
            if step and step % 512 == 0:
                state, _ = env.reset()
                if env.hard_reset_count != 1:
                    raise AssertionError("Scenario changed inside four carried blocks")
            # HRR features are 44 pressure + 60 action + 4 rate + 2 reserve + 52 ref.
            # Remove only the two reserve coordinates to evaluate the frozen HR.
            hr_state = np.concatenate((state[:108], state[110:]))
            action = model.predict(hr_state, evaluate=True)
            state, reward, done, truncated, info = env.step(action)
            if done or truncated or not np.isfinite(state).all() or not np.isfinite(reward):
                raise AssertionError(f"Invalid frozen rollout in {scenario['name']}")
            sample = sampler.sample()
            for key in ("native_state", "valve_debug", "mass_flowrate"):
                arrays[key][step] = sample[key]
            raw = np.asarray(info["ctrl_input"]).reshape(-1, 6)[0]
            arrays["physical_command"][step] = .8 + .2 * raw
            arrays["observed_pressure"][step] = env.curr_full_press
            arrays["reference"][step] = [info["obs"]["ref_act_pos"], info["obs"]["ref_act_neg"]]
        if not np.isclose(env.t, 2048 / 50):
            raise AssertionError("Reference time did not carry through four blocks")
    finally:
        env.close()
    physics = analyze_arrays({key: arrays[key][None] for key in (
        "native_state", "valve_debug", "mass_flowrate", "physical_command")})
    metrics = {}
    for i, name in enumerate(("positive", "negative")):
        error = arrays["observed_pressure"][:, i + 2] - arrays["reference"][:, i]
        f, power = welch(error, fs=50., nperseg=1024, detrend="linear")
        mask = (f >= 2.) & (f <= 20.)
        metrics[name] = {"tracking_rmse_kpa": float(np.sqrt(np.mean(error ** 2))),
                         "tracking_bias_kpa": float(np.mean(error)),
                         "tracking_error_std_kpa": float(np.std(error)),
                         "error_band_2_20hz_rms_kpa": float(np.sqrt(np.sum(power[mask]) * (f[1] - f[0])))}
    return {"scenario": scenario, "passed": physics["passed"], "initial": initial,
            "steps": steps, "hard_resets": 1, "metrics": metrics, "feasibility": physics,
            "reference_array_sha256": __import__("hashlib").sha256(arrays["reference"].tobytes()).hexdigest()}


def validate(library, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"passed": False, "source_config_sha256": sha256(SOURCE),
              "dr_profile_sha256": sha256(DR_PROFILE),
              "implementation_hashes": {path: sha256(ROOT / path) for path in IMPLEMENTATIONS},
              "randomized_library_sha256": sha256(library), "checks": {}}
    try:
        baseline = build_config("baseline", library=library)
        randomized = build_config("dr", library=library)
        bank_path = ROOT / baseline["reference"]["kwargs"]["trajectory_npz"]
        with np.load(bank_path, allow_pickle=False) as bank:
            pressures = bank["pressure"]
            report["reference_bank"] = {
                "path": str(bank_path), "sha256": sha256(bank_path),
                "members": int(pressures.shape[0]),
                "all_member_target_min_kpa": pressures.min(axis=(0, 1)).tolist(),
                "all_member_target_max_kpa": pressures.max(axis=(0, 1)).tolist(),
                "screened_member": 0,
                "reachability_status": "Not certified for all members and randomized conditions."}
        scenarios = randomized["training_dynamics_randomization"]["scenarios"]
        nominal = next(s for s in scenarios if s["name"] == "nominal")
        nominal_config = one_scenario(randomized, nominal)
        actions = np.random.default_rng(61026).uniform(0, 1, (128, 6))
        base_values = fixed_action_rollout(baseline, actions)
        random_values = fixed_action_rollout(nominal_config, actions)
        parity = max(float(np.max(np.abs(a - b))) for a, b in zip(base_values, random_values))
        if parity != 0.:
            raise AssertionError(f"Nominal HRR environment parity failed: {parity}")
        report["checks"]["nominal_environment_parity"] = {
            "passed": True, "max_abs_error": parity, "precondition_seconds": 30.,
            "physical_steps": len(actions), "compared": ["raw policy state", "native state", "reward"]}
        for variant, cfg in (("baseline", baseline), ("dr", nominal_config)):
            report["checks"][f"{variant}_sac_update"] = update_smoke(cfg, output.parent)
        hashes = [report["checks"][f"{variant}_sac_update"]["initial_actor_sha256"]
                  for variant in ("baseline", "dr")]
        if hashes[0] != hashes[1]:
            raise AssertionError("Baseline and DR fresh actors have different initialization")
        report["checks"]["same_initial_actor"] = {"passed": True, "sha256": hashes[0]}
        frozen = frozen_hr_model()
        frozen_hash = policy_hash(frozen)
        report["frozen_hr_checkpoint"] = {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT)}
        scenario_results = []
        for scenario in scenarios:
            result = scenario_rollout(randomized, frozen, scenario)
            scenario_results.append(result)
            print(json.dumps({"scenario": scenario["name"], "passed": result["passed"],
                              "metrics": result["metrics"]}), flush=True)
            if not result["passed"]:
                raise AssertionError(f"Native physical invariants failed: {scenario['name']}")
        if policy_hash(frozen) != frozen_hash:
            raise AssertionError("Frozen baseline actor changed during evaluation")
        if len({row["reference_array_sha256"] for row in scenario_results}) != 1:
            raise AssertionError("Scenario comparison did not use identical reference samples")
        report["checks"]["scenario_physical_invariants"] = {
            "passed": True, "results": scenario_results,
            "total_policy_steps": sum(row["steps"] for row in scenario_results)}
        report["scope"] = [
            "Reference bank member 0, existing frozen HR actor, 2048 steps per condition.",
            "Physical invariants are screened on recorded native 50 Hz states, not every 100 us step.",
            "Supply operating-envelope violations are reported separately; tracking quality has no arbitrary gate threshold.",
            "These checks do not certify reference reachability over all members, globally stable policies, or real-device randomization coverage.",
            "The two fresh HRR smoke actors receive one SAC update each and are discarded; no training checkpoint is retained.",
        ]
        report["passed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["implementation_hashes"] = {path: sha256(ROOT / path) for path in IMPLEMENTATIONS}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=EXPERIMENT / "build/lib3_type1_randomized.so")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "preflight.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    saved_random = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    saved_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = validate(args.library.resolve(), args.output.resolve())
        print(json.dumps({"passed": result["passed"], "output": str(args.output),
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    finally:
        random.setstate(saved_random)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        torch.set_num_threads(saved_threads)
