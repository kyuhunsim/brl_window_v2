"""Resume valid rolling SAC weights with an explicitly older replay snapshot.

Recover reference/dynamics scheduling, validate the replay's physical history,
then start a fresh prepared rollout. Missing recent replay and stochastic/alpha
optimizer state are recorded. Neither configuration nor the SAC algorithm is
changed. This recovery does not reproduce uninterrupted training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for package in ("pneu_env", "pneu_ref", "pneu_rl", "pneu_utils"):
    sys.path.insert(0, str(ROOT / package / "src"))

import numpy as np
import torch
import yaml

from resume_lib3_hrr_comparison import physical_schedule, restore_physical_rollout, save_json
from train_lib3_hrr_comparison import make_training_components, policy_hash, sha256
from train_screening import seed_everything
from pneu_rl.sac_with_loss import SAC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "dr"), required=True)
    parser.add_argument("--replay-episode", type=int, default=1800)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--name")
    parser.add_argument("--recovery-dir", type=Path,
                        default=ROOT / "pneu_rl/exp/lib3_hrr_dr_20261026/recovery_20261027_0015")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    name = args.name or f'261026_lib3_pressurecal_type1_HRR{"_DR" if args.variant == "dr" else ""}_smooth_seed0'
    if Path(name).name != name:
        parser.error("name must be one model folder")
    folder = ROOT / "pneu_rl/models" / name
    cfg = yaml.safe_load((folder / "cfg.yaml").read_text())
    if cfg["episodes"] != args.episodes or cfg["model"]["horizon"] != 512:
        raise ValueError("Saved total episodes/horizon do not match recovery")
    provenance = cfg["plant_provenance"]
    for path, expected in (
        (Path(cfg["obs"]["library_path"]), provenance["simulator_library_sha256"]),
        (ROOT / cfg["reference"]["kwargs"]["trajectory_npz"], provenance["reference_bank_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen simulator/reference has changed: {path}")
    with (folder / "infos.pkl").open("rb") as handle:
        infos = pickle.load(handle)
    latest = max(infos)
    if not 0 < args.replay_episode <= latest < args.episodes:
        raise ValueError("Require 0 < replay episode <= latest episode < target episode")
    if args.replay_episode not in infos:
        raise RuntimeError("Saved replay checkpoint episode is absent from infos")

    checkpoint = folder / "model.pth"
    replay_file = folder / "buffer.pkl"
    checkpoint_hash = sha256(checkpoint)
    replay_mtime = replay_file.stat().st_mtime
    seed_everything(int(cfg["train_seed"]))
    env = make_training_components(cfg, scenario_log=folder / "dynamics_randomization.jsonl")
    model = SAC(env=env, **cfg["model"])
    model.set_logger(name)
    model.logger.infos = infos
    model.last_epi = latest
    model.total_steps = int(infos[latest]["steps"])
    if model.total_steps != latest * model.horizon:
        raise RuntimeError("Latest episode is not a complete saved horizon")
    model.load_model(str(checkpoint), evaluate=False)
    model.is_model_loaded = True
    if not torch.equal(model.log_alpha.exp(), infos[latest]["alpha"]):
        raise RuntimeError("Rolling weights and latest infos do not have matching alpha")
    expected_updates = model.total_steps - cfg["model"]["start_epi"] * model.horizon
    for optimizer in (model.policy_optim, model.critic_optim):
        steps = {float(item["step"]) for item in optimizer.state.values()}
        if steps != {float(expected_updates)}:
            raise RuntimeError("Rolling optimizer steps disagree with latest saved episode")
    actor_hash = policy_hash(model)
    with replay_file.open("rb") as handle:
        model.buffer.buffer = pickle.load(handle)
    if len(model.buffer) != min(args.replay_episode * model.horizon, cfg["model"]["buffer_size"]):
        raise RuntimeError("Replay snapshot has an unexpected number of transitions")

    # Verify the old replay against its own episode, never describe it as the
    # newer policy's missing physical state. No replay entries or updates added.
    replay_validation = restore_physical_rollout(
        env, model, cfg, args.replay_episode, folder / "dynamics_randomization.jsonl")
    latest_reset_count, _, latest_completed_blocks = physical_schedule(latest, cfg)
    randomized = bool(cfg.get("training_dynamics_randomization", {}).get("enabled"))
    records = None
    if randomized:
        records = [json.loads(line) for line in
                   (folder / "dynamics_randomization.jsonl").read_text().splitlines()]
        if len(records) != latest_reset_count:
            raise RuntimeError("Existing DR schedule is not aligned with this recovery segment")
    for index in range(env.hard_reset_count, latest_reset_count):
        env.goal.ref.time_reset()
        if randomized:
            env._select_dynamics()
            if env.current_dynamics_scenario != records[index]["scenario"]:
                raise RuntimeError("Historical DR sequence changed while advancing schedule")
    env.hard_reset_count = latest_reset_count
    env.request_hard_reset()
    saved_dr_log = env._dr_log_path if randomized else None
    if randomized:
        env._dr_log_path = None  # Append only after all validation passes.
    prepared_state, prepared_info = env.reset()
    if prepared_state.shape != (162,) or not np.isfinite(prepared_state).all():
        raise RuntimeError("Fresh prepared HRR observation is invalid")
    action = model.predict(prepared_state, evaluate=True)
    if not np.isfinite(action).all() or np.any(action < 0) or np.any(action > 1):
        raise RuntimeError("Restored actor produced an invalid deterministic action")
    if actor_hash != policy_hash(model) or checkpoint_hash != sha256(checkpoint):
        raise RuntimeError("Weights changed during recovery validation")

    # SAC.train calls reset at its first episode. Return this already prepared
    # state once so that this call does not consume a zero-length carry block.
    normal_reset = env.reset

    def first_training_reset():
        env.reset = normal_reset
        return prepared_state, prepared_info

    env.reset = first_training_reset
    seed_everything(int(cfg["train_seed"]))
    started = time.time()
    report = {
        "variant": args.variant, "model": name, "dry_run": args.dry_run,
        "started_unix": started, "checkpoint_episode": latest,
        "checkpoint_sha256": checkpoint_hash, "restored_actor_sha256": actor_hash,
        "target_episode": args.episodes, "additional_episodes": args.episodes - latest,
        "restored_policy_steps": model.total_steps,
        "replay_checkpoint_episode": args.replay_episode,
        "replay_modified_unix": replay_mtime, "replay_entries": len(model.buffer),
        "missing_recent_replay_episodes": latest - args.replay_episode,
        "missing_recent_replay_transitions": (latest - args.replay_episode) * model.horizon,
        "older_replay_validation": replay_validation,
        "fresh_physical_reset": {
            "first_training_episode": latest + 1,
            "previous_completed_carry_blocks": latest_completed_blocks,
            "hard_reset_count": env.hard_reset_count,
            "carry_block_index": env.carry_block_index,
            "preparation_seconds": env.precondition_seconds,
            "pressure_kpa": env.curr_full_press.tolist(),
            "reference_kpa": env._current_actuator_reference().tolist(),
            "reference_selection_count": env.goal.ref._episode_index,
            "active_reference_member": int(env.goal.ref._active_index),
            "policy_time_seconds": env.t,
            "native_time_seconds": float(env.obs.lib.get_time_c()),
            "scenario": getattr(env, "current_dynamics_scenario", None),
            "preparation_excluded_from_replay_and_reward": True,
        },
        "initial_deterministic_action": action.tolist(),
        "training_rng_seed": int(cfg["train_seed"]), "torch_threads": args.threads,
        "bit_identical_uninterrupted_training": False,
        "not_recovered": ["latest_physical_state", "recent_unsaved_replay_transitions",
                          "alpha_optimizer_moments", "policy_and_replay_sampling_rng"],
        "completed": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    report_path = args.recovery_dir / f'{args.variant}_{"dry_run" if args.dry_run else "resume"}.json'
    save_json(report_path, report)
    if args.dry_run:
        env.close()
        return

    if randomized:
        env._dr_log_path = saved_dr_log
        record = {
            "hard_reset_count": env.hard_reset_count,
            "scenario_index_in_rollout_sequence": env._dr_draw_count - 1,
            "scenario": env.current_dynamics_scenario,
            "preparation_seconds": env.precondition_seconds,
            "policy_start_pressure_kpa": env.curr_full_press.tolist(),
            "policy_start_reference_kpa": env._current_actuator_reference().tolist(),
            "initial_tracking_error_kpa": (env.curr_full_press[2:]
                                           - env._current_actuator_reference()).tolist(),
            "effective_delay_steps": round(env.obs.freq * env.obs.delay),
            "native_time_seconds": float(env.obs.lib.get_time_c()),
            "recovery_first_episode": latest + 1,
            "fresh_reset_after_unsaved_physical_state": True,
        }
        with saved_dr_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    wall_path = folder / "wall_time.json"
    wall = json.loads(wall_path.read_text())
    wall.setdefault("recovery_segments", []).append(report)
    wall.update(completed=False, active_recovery_started_unix=started)
    save_json(wall_path, wall)
    try:
        model.train(episode=args.episodes - latest)
        if max(model.logger.infos) != args.episodes:
            raise RuntimeError("Training returned before target episode")
        # The shared trainer saves replay every 100 LOCAL episodes. A 169/170
        # episode recovery must explicitly save the final 2000-episode replay.
        model.buffer.save_buffer(model.logger.buffer_path)
        report["completed"] = True
        report["final_replay_checkpoint_episode"] = args.episodes
    finally:
        finished = time.time()
        report.update(finished_unix=finished, elapsed_seconds=finished - started,
                      final_episode=max(model.logger.infos),
                      final_policy_steps=model.total_steps)
        wall.update(finished_unix=finished,
                    elapsed_seconds=finished - wall["started_unix"],
                    elapsed_includes_interruption_gaps=True,
                    completed=report["completed"], policy_steps=model.total_steps)
        save_json(report_path, report)
        save_json(wall_path, wall)
        env.close()


if __name__ == "__main__":
    main()
