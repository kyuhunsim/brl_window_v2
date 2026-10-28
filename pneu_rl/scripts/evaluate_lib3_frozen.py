#!/usr/bin/env python3
"""Frozen lib3 policy evaluation: all-open preparation -> hold -> bank.

No replay, optimizer, or training loop is constructed. See
README_lib3_frozen_eval.txt for the four-run Real protocol.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
for package in ("pneu_env", "pneu_rl", "pneu_ref", "pneu_utils"):
    sys.path.insert(0, str(ROOT / package / "src"))

import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from pneu_rl.networks import SACPolicy
from pneu_ref.hold_then_bank_ref import HoldThenBankRef

COMMANDS = ("ctrl_ch_pos", "ctrl_ch_neg", "ctrl_act_pos_in", "ctrl_act_pos_out",
            "ctrl_act_neg_in", "ctrl_act_neg_out")
PRESSURES = ("press_pos", "press_neg", "act_pos_press", "act_neg_press")
SOURCE_FILES = (
    "pneu_rl/scripts/evaluate_lib3_frozen.py", "pneu_rl/src/pneu_rl/networks.py",
    "pneu_rl/src/pneu_rl/deployment.py", "pneu_env/src/pneu_env/env3.py",
    "pneu_env/src/pneu_env/env3_all_open.py", "pneu_env/src/pneu_env/sim3.py",
    "pneu_env/src/pneu_env/real/real9.py", "pneu_env/src/pneu_env/real/real_act.py",
    "pneu_ref/src/pneu_ref/pneu_ref.py", "pneu_ref/src/pneu_ref/hold_then_bank_ref.py",
    "pneu_ref/src/pneu_ref/optimized_trajectory_ref.py",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def actor_hash(policy):
    h = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().numpy()
        h.update(key.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def resolve_resource(model_dir, configured, bundled):
    path = Path(configured) if configured else None
    # A transferred bundle's own resource takes precedence over a stale PC path.
    candidates = [model_dir / bundled]
    if path is not None:
        candidates += [path] if path.is_absolute() else [model_dir / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing {bundled}; checked {candidates}")


class FrozenActor:
    def __init__(self, weights, config, *, action_space, state_scale, seed):
        self.state_scale = np.asarray(state_scale, dtype=np.float32)
        if np.any(~np.isfinite(self.state_scale)) or np.any(self.state_scale <= 0):
            raise ValueError("Invalid policy observation scale")
        self.policy = SACPolicy(len(self.state_scale), action_space.shape[0],
            hidden_dim=config.get("hidden_dim", 64), action_space=action_space,
            log_std_min=config.get("log_std_min", -10),
            log_std_max=config.get("log_std_max", 1))
        self.policy.load_state_dict(weights, strict=True)
        self.policy.eval(); self.policy.requires_grad_(False)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def action(self, state, mode):
        if mode not in ("deterministic", "stochastic"):
            raise ValueError(mode)
        raw = np.asarray(state, dtype=np.float32)
        if raw.shape != self.state_scale.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid policy observation")
        normalized = raw / self.state_scale
        with torch.no_grad():
            mu, log_std = self.policy(torch.from_numpy(normalized).unsqueeze(0))
            epsilon = (torch.randn(mu.shape, generator=self.generator)
                       if mode == "stochastic" else torch.zeros_like(mu))
            latent = mu + log_std.exp() * epsilon
            action = self.policy.action_scale * torch.tanh(latent) + self.policy.action_bias
            center = self.policy.action_scale * torch.tanh(mu) + self.policy.action_bias
        audit = {"state": raw.copy(), "policy_input": normalized.copy(),
                 "pre_tanh_mu": mu[0].numpy(), "pre_tanh_log_std": log_std[0].numpy(),
                 "epsilon": epsilon[0].numpy(), "action": action[0].numpy(),
                 "deterministic_action": center[0].numpy()}
        if not all(np.isfinite(x).all() for x in audit.values()):
            raise ValueError("Non-finite actor output")
        return audit["action"].copy(), audit


def record_preparation(env, rows):
    """Capture reset exchanges too; caller distinguishes last N all-open steps."""
    original = env.obs.observe
    def observe(action, goal):
        start = time.monotonic()
        observation, info = original(action, goal)
        row = {"exchange_index": len(rows), "backend_time": float(observation[0]),
               "host_monotonic_end": time.monotonic(),
               "exchange_duration_ms": (time.monotonic() - start) * 1000}
        row.update(zip(PRESSURES, np.asarray(observation[1:5], float)))
        physical = env.obs.scale_low + (env.obs.scale_high - env.obs.scale_low) * np.asarray(action)
        row.update(zip(COMMANDS, physical.reshape(-1)))
        observation_info = info.get("Observation", info)
        for k in range(1, 7):
            row[f"flowrate{k}"] = observation_info.get(f"flowrate{k}", float("nan"))
        rows.append(row)
        return observation, info
    env.obs.observe = observe
    try:
        return env.reset()
    finally:
        env.obs.observe = original


def collect(env, actor, ref, mode, output, manifest):
    rows, preparation, traces = [], [], []
    status, error, close_error = "completed", None, None
    fieldnames = ["step", "time", "reference_query_time", "reference_current_time",
        "bank_time", "phase", "sensor_time", "host_monotonic_start", "host_monotonic_end",
        "policy_inference_ms", "environment_step_ms", "host_cycle_ms", "reward",
        *PRESSURES, "ref_act_pos", "ref_act_neg", *COMMANDS,
        *["echo_" + c for c in COMMANDS], *[f"flowrate{k}" for k in range(1, 7)]]
    output = Path(output)
    with (output / "rollout.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader(); f.flush()
        try:
            state, info = record_preparation(env, preparation)
            manifest["initial_pressure_kpa"] = env.curr_full_press.tolist()
            manifest["initial_reference_kpa"] = list(ref.get_goal(0))
            manifest["preparation_exchanges"] = len(preparation)
            manifest["prepared_steps"] = env.precondition_steps
            steps = round(ref.max_time * env.obs.freq)
            for i in range(steps):
                start = time.monotonic(); query_time = float(env.t)
                action, audit = actor.action(state, mode)
                physical = env.obs.scale_low + (env.obs.scale_high - env.obs.scale_low) * action
                if np.any(physical < .8 - 1e-6) or np.any(physical > 1 + 1e-6):
                    raise ValueError("Command outside calibrated [0.8, 1.0] domain")
                audit.update(physical_command=physical.copy(), completed=False,
                             next_state=np.full_like(audit["state"], np.nan))
                traces.append(audit)
                after_actor = time.monotonic()
                next_state, reward, terminated, truncated, info = env.step(action)
                end = time.monotonic()
                if not np.isfinite(next_state).all() or not np.isfinite(reward):
                    raise ValueError("Non-finite response/reward")
                audit["next_state"] = np.asarray(next_state, np.float32).copy()
                audit["completed"] = True
                o = info["obs"]
                # Current reference index in existing PneuRef is t_query+dt after
                # the preview buffer warms. Hold duration covers that warm-up.
                current_time = (i + 1) / env.obs.freq
                row = dict(step=i, time=current_time, reference_query_time=query_time,
                    reference_current_time=current_time,
                    bank_time=max(0.0, min(ref.bank_seconds, current_time-ref.hold_seconds)),
                    phase="hold" if current_time <= ref.hold_seconds else "bank",
                    sensor_time=float(o["curr_time"]),
                    host_monotonic_start=start, host_monotonic_end=end,
                    policy_inference_ms=(after_actor-start)*1000,
                    environment_step_ms=(end-after_actor)*1000, host_cycle_ms=(end-start)*1000,
                    reward=float(reward), ref_act_pos=float(o["ref_act_pos"]),
                    ref_act_neg=float(o["ref_act_neg"]))
                row.update(zip(PRESSURES, map(float, env.curr_full_press)))
                row.update(zip(COMMANDS, map(float, physical)))
                echo_keys = ("ctrl_pos", "ctrl_neg", "ctrl_act_pos_in", "ctrl_act_pos_out",
                             "ctrl_act_neg_in", "ctrl_act_neg_out")
                row.update({"echo_"+name: float(o.get(key, np.nan))
                            for name, key in zip(COMMANDS, echo_keys)})
                row.update({f"flowrate{k}": float(o.get(f"flowrate{k}", np.nan)) for k in range(1,7)})
                writer.writerow(row); rows.append(row)
                if (i+1) % round(env.obs.freq) == 0:
                    f.flush()
                state = next_state
                if terminated or truncated:
                    raise RuntimeError("Environment ended before evaluation duration")
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
        except Exception:
            status, error = "failed", traceback.format_exc()
        finally:
            # Close before disk compression/plotting; Real publishes all-open.
            try:
                env.close()
            except Exception:
                close_error = traceback.format_exc(); status = "failed"
            f.flush()
    if preparation:
        with (output/"preparation.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(preparation[0]))
            writer.writeheader(); writer.writerows(preparation)
    arrays = {key: np.stack([x[key] for x in traces]) for key in traces[0]} if traces else {}
    np.savez_compressed(output/"policy_trace.npz", **arrays, state_scale=actor.state_scale)
    manifest.update(status=status, error=error, close_error=close_error,
        completed_transitions=len(rows), attempted_transitions=len(traces),
        actor_sha256_after=actor_hash(actor.policy))
    manifest["actor_unchanged"] = manifest["actor_sha256_before"] == manifest["actor_sha256_after"]
    if not manifest["actor_unchanged"]:
        manifest["status"] = "failed"
    write_json(output/"manifest.json", manifest)
    return rows


def summarize(rows, ref, output, show=False):
    import pandas as pd
    from scipy.signal import welch
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    windows = {"whole": np.ones(len(df), bool), "hold_all": df.time <= ref.hold_seconds,
        "hold_last10": (df.time > max(0, ref.hold_seconds-10)) & (df.time <= ref.hold_seconds),
        "bank_all": df.time > ref.hold_seconds,
        "bank_after5": df.time > ref.hold_seconds+5}
    metrics = {}
    for label, mask in windows.items():
        d = df.loc[mask]
        if len(d) == 0:
            continue
        result = {"samples": len(d), "axes": {}, "commands": {}}
        for axis in ("pos", "neg"):
            pressure = d[f"act_{axis}_press"].to_numpy()
            error = pressure - d[f"ref_act_{axis}"].to_numpy()
            m = dict(rmse_kpa=float(np.sqrt(np.mean(error**2))), bias_kpa=float(error.mean()),
                error_std_kpa=float(error.std()), pressure_peak_to_peak_kpa=float(np.ptp(pressure)))
            t = d.sensor_time.to_numpy()
            if len(t) >= 64 and np.all(np.diff(t) > 0):
                grid = np.arange(t[0], t[-1], .02)
                if len(grid) >= 64:
                    freq, psd = welch(np.interp(grid, t, pressure), fs=50,
                                     nperseg=min(512,len(grid)), detrend="linear")
                    selected = (freq >= 2) & (freq <= 20)
                    m["pressure_rms_2_20hz_kpa"] = float(np.sqrt(np.sum(psd[selected])*(freq[1]-freq[0])))
            result["axes"][axis] = m
        for col in COMMANDS:
            u = d[col].to_numpy()
            result["commands"][col] = dict(mean_abs_delta_over_full_range=(float(np.abs(np.diff(u)).mean()/.2) if len(u)>1 else None),
                endpoint_flips=int(np.sum(((u[:-1]<=.81)&(u[1:]>=.99))|((u[:-1]>=.99)&(u[1:]<=.81)))))
        metrics[label] = result
    write_json(Path(output)/"metrics.json", metrics)
    # Keep Quick Viz's exact plotting convention for pressure and six commands.
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    from pneu_rl.plotting import plot_lib3_pressure_and_controls
    plot_lib3_pressure_and_controls(df, Path(output)/"quick_viz", show=show)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model folder name or path (contains cfg.yaml)")
    parser.add_argument("--checkpoint", default="model.pth", help="Relative to model folder, or absolute path")
    parser.add_argument("--obs", choices=("sim", "real"), default="sim")
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--hold-seconds", type=float, default=20.)
    parser.add_argument("--bank-seconds", type=float, default=40.)
    parser.add_argument("--preparation-seconds", type=float, default=30.)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, help="New, empty output folder; default exp/<model>/<run>")
    parser.add_argument("--show", action="store_true", help="Show Quick Viz figures after shutdown")
    parser.add_argument("--dry-run", action="store_true", help="Validate files/actor without any Sim or Real backend")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    model_dir = Path(args.model).expanduser()
    if not model_dir.is_dir():
        model_dir = ROOT/"pneu_rl/models"/args.model
    model_dir = model_dir.resolve()
    cfg = yaml.safe_load((model_dir/"cfg.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    freq = float(cfg["obs"]["freq"])
    if freq != 50 or cfg["env"].get("action_chunk_steps",1) != 1 or cfg.get("pred") is not None:
        raise ValueError("This protocol requires 50 Hz, K=1, no predictor")
    if cfg.get("initialization",{}).get("mode") != "pump_on_all_open":
        raise ValueError("This protocol requires an all-open prepared model")
    if not cfg["obs"].get("scale",False) or cfg["obs"].get("scale_low",.8) != .8 or cfg["obs"].get("scale_high",1.) != 1.:
        raise ValueError("Expected normalized actions mapped to physical [0.8,1.0]")
    if args.hold_seconds < (cfg["env"].get("num_pred",15)+1)/freq:
        raise ValueError("Hold must fill the reference preview buffer before bank begins")
    for seconds in (args.hold_seconds,args.bank_seconds,args.preparation_seconds):
        if not np.isfinite(seconds) or seconds < 0 or not np.isclose(seconds*freq,round(seconds*freq)):
            raise ValueError("Durations must be finite multiples of one 50 Hz step")
    if args.preparation_seconds*freq < (cfg["env"].get("num_prev",10)+1+round(cfg["obs"].get("delay",0)*freq)):
        raise ValueError("Preparation must fill observation and delay history")
    reference_path = resolve_resource(model_dir, cfg.get("reference",{}).get("kwargs",{}).get("trajectory_npz"), "reference.npz")
    ref = HoldThenBankRef(trajectory_npz=reference_path, trajectory_index=args.trajectory_index,
                         hold_seconds=args.hold_seconds, bank_seconds=args.bank_seconds)
    checkpoint = Path(args.checkpoint)
    checkpoint = (checkpoint if checkpoint.is_absolute() else model_dir/checkpoint).resolve()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["policy_state_dict"]
    state_dim = weights["linear_1.weight"].shape[1]
    action_dim = weights["linear_mean.weight"].shape[0]
    if action_dim != 6:
        raise ValueError("Expected six direct valve commands")
    # Verify checkpoint shape/config before any connection or commands.
    space = Box(low=float(cfg["env"].get("action_low",0)), high=float(cfg["env"].get("action_high",1)), shape=(6,), dtype=np.float32)
    actor = FrozenActor(weights, cfg["model"], action_space=space, state_scale=np.ones(state_dim), seed=args.seed)
    if args.obs == "sim":
        library = resolve_resource(model_dir, cfg["obs"].get("library_path"), "simulator.so")
        cfg["obs"]["library_path"] = str(library)
    else:
        library = None
    cfg["initialization"]["precondition_seconds"] = args.preparation_seconds
    cfg["env"]["verbose"] = False
    name = f"{datetime.now(ZoneInfo('Asia/Seoul')):%Y%m%d_%H%M%S_%f}_frozen_{args.obs}_{args.mode}_seed{args.seed}"
    output = (args.output or ROOT/"pneu_rl/exp"/model_dir.name/name).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source in SOURCE_FILES:
        target = output/"source_snapshot"/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source,target); sources[source] = sha256(target)
    try:
        git_head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:
        git_head = None
    manifest = dict(status="validated" if args.dry_run else "starting", model=str(model_dir),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        config_sha256=sha256(model_dir/"cfg.yaml"), actor_sha256_before=actor_hash(actor.policy),
        obs=args.obs, mode=args.mode, seed=args.seed, learning_enabled=False,
        policy_weight_updates=0, reference=ref.metadata(), frequency_hz=freq,
        preparation_seconds=args.preparation_seconds, state_dim=state_dim,
        simulator_path=str(library) if library else None,
        simulator_sha256=sha256(library) if library else None,
        source_sha256=sources, git_head=git_head, arguments=vars(args) | {"output":str(output)},
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        notes=["No optimizer or replay is instantiated; actor is frozen in both modes.",
               "pre_tanh_log_std describes latent Gaussian, not physical command std.",
               "deterministic_action is tanh(mu), not E[action|state].",
               "sensor_time is backend time, not proven sensor acquisition time.",
               "ctrl columns are requested physical commands; echo is not measured valve opening.",
               "Reset/exchange records are in preparation.csv; excluded from metrics.",
               "Existing reference history semantics preserved; hold fills initial preview buffer."])
    manifest["arguments"]["output"] = str(output)
    (output/"resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy2(model_dir/"cfg.yaml", output/"source_config.yaml")
    shutil.copy2(reference_path, output/"reference.npz")
    write_json(output/"manifest.json",manifest)
    print(f"Model: {model_dir.name}\nActor: {manifest['actor_sha256_before']}\n"
          f"{args.obs} / {args.mode} / seed={args.seed}\n"
          f"Prepare {args.preparation_seconds:g}s -> hold {args.hold_seconds:g}s -> bank {args.bank_seconds:g}s\n"
          f"Output: {output}", flush=True)
    if args.dry_run:
        print("File validation complete; no backend opened."); return 0
    from pneu_env.sim3 import PneuSim
    from pneu_env.real.real9 import PneuReal
    from pneu_rl.deployment import build_deployment_env
    obs = env = None
    try:
        obs = PneuSim(**cfg["obs"]) if args.obs=="sim" else PneuReal(**cfg["obs"])
        env = build_deployment_env(cfg, obs, ref, None)
        scale = env.policy_state_scale if cfg["model"].get("normalize_observation",False) else np.ones(env.observation_space.shape[0])
        actor = FrozenActor(weights, cfg["model"], action_space=env.action_space, state_scale=scale, seed=args.seed)
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        try:
            if env is not None:
                env.close()
            elif obs is not None and args.obs=="real":
                obs.safe_open()
        except Exception:
            manifest["close_error"] = traceback.format_exc()
        write_json(output/"manifest.json",manifest)
        raise
    rows = collect(env,actor,ref,args.mode,output,manifest)
    # Environment has closed even if plotting/importing scipy fails.
    try:
        metrics = summarize(rows,ref,output,args.show)
        for label in ("hold_last10","bank_all"):
            if label in metrics:
                a=metrics[label]["axes"]
                print(f"{label}: RMSE POS {a['pos']['rmse_kpa']:.4f}, NEG {a['neg']['rmse_kpa']:.4f} kPa")
    except Exception:
        manifest["analysis_error"] = traceback.format_exc()
        write_json(output/"manifest.json",manifest)
        print(manifest["analysis_error"],file=sys.stderr)
        return 1
    print(f"Status: {manifest['status']}; actor unchanged: {manifest['actor_unchanged']}")
    if manifest.get("error"):
        print(manifest["error"],file=sys.stderr)
    return 0 if manifest["status"]=="completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
