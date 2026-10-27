"""Recover the matched HRR runs from a complete model/replay checkpoint.

Reconstruct the last physical rollout from its saved actions before learning.
This does not make training bit-identical: policy/replay RNG and alpha-optimizer
moments were not checkpointed. Actor/critic optimizers and learned alpha are
restored, and reconstruction must match every saved transition before resuming.
"""
from __future__ import annotations

import argparse
from itertools import islice
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

from train_lib3_hrr_comparison import make_training_components, policy_hash, sha256
from train_screening import seed_everything
from pneu_rl.sac_with_loss import SAC


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def physical_schedule(checkpoint_episode, cfg):
    """Reproduce carried reset counts, including reset after fixed warm-up."""
    blocks = int(cfg["env"]["episode_carry_blocks"])
    if blocks <= 0 or not cfg["env"]["episode_carry_state"]:
        raise ValueError("Recovery expects the saved finite carried-block schedule")
    count = 0
    block = blocks
    start_episode = None
    for episode in range(1, checkpoint_episode + 1):
        forced = (cfg["model"].get("reset_after_warmup", False)
                  and episode - 1 == cfg["model"]["start_epi"])
        if block >= blocks or forced:
            count += 1
            block = 1
            start_episode = episode
        else:
            block += 1
    return count, start_episode, block


def restore_physical_rollout(env, model, cfg, episode, scenario_log):
    reset_count, first_episode, completed_blocks = physical_schedule(episode, cfg)
    horizon = int(model.horizon)
    transitions_needed = (episode - first_episode + 1) * horizon
    transitions = list(islice(model.buffer.buffer, len(model.buffer) - transitions_needed, None))
    if len(transitions) != transitions_needed:
        raise RuntimeError("Replay does not contain the complete last physical rollout")

    reference = env.goal.ref
    if reference._episode_index != 2:
        raise RuntimeError("Unexpected reference constructor reset count")
    for _ in range(reset_count - 1):
        reference.time_reset()
    env.hard_reset_count = reset_count - 1

    randomized = bool(cfg.get("training_dynamics_randomization", {}).get("enabled"))
    saved_log_path = None
    if randomized:
        records = [json.loads(line) for line in scenario_log.read_text().splitlines()]
        if len(records) < reset_count:
            raise RuntimeError("Incomplete dynamics schedule for checkpoint recovery")
        saved_log_path = env._dr_log_path
        env._dr_log_path = None  # Do not duplicate historical preparation entries.
        for index in range(reset_count - 1):
            env._select_dynamics()
            if env.current_dynamics_scenario != records[index]["scenario"]:
                raise RuntimeError("Saved dynamics schedule disagrees with seeded replay")
    state, _ = env.reset()
    if randomized and env.current_dynamics_scenario != records[reset_count - 1]["scenario"]:
        raise RuntimeError("Restored last dynamics scenario differs from checkpoint rollout")

    max_state_error = 0.0
    max_reward_error = 0.0
    restored_episode_reward = 0.0
    episode_reward_errors = []
    for index, transition in enumerate(transitions):
        if index and index % horizon == 0:
            state, _ = env.reset()
        obs, goal, action, reward, next_state, mask = transition
        expected_state = np.r_[obs, goal]
        max_state_error = max(max_state_error, float(np.max(np.abs(state - expected_state))))
        if not np.allclose(state, expected_state, atol=1e-7, rtol=1e-10):
            raise RuntimeError(f"Recovered input state differs at replay transition {index}")
        state, actual_reward, terminated, truncated, _ = env.step(np.asarray(action).copy())
        max_state_error = max(max_state_error, float(np.max(np.abs(state - next_state))))
        max_reward_error = max(max_reward_error, abs(float(actual_reward) - float(reward)))
        if not np.allclose(state, next_state, atol=1e-7, rtol=1e-10):
            raise RuntimeError(f"Recovered next state differs at replay transition {index}")
        if not np.isclose(actual_reward, reward, atol=1e-7, rtol=1e-10):
            raise RuntimeError(f"Recovered reward differs at replay transition {index}")
        if terminated or truncated or mask != 1:
            raise RuntimeError("Unexpected terminal transition in carried rollout")
        restored_episode_reward += actual_reward
        if (index + 1) % horizon == 0:
            original_episode = first_episode + index // horizon
            expected_reward = float(model.logger.infos[original_episode]["reward"])
            error = abs(restored_episode_reward - expected_reward)
            episode_reward_errors.append(error)
            if not np.isclose(restored_episode_reward, expected_reward, atol=1e-6, rtol=1e-10):
                raise RuntimeError("Replay is not aligned with the recorded checkpoint episodes")
            restored_episode_reward = 0.0
    if randomized:
        env._dr_log_path = saved_log_path
    if env.hard_reset_count != reset_count or env.carry_block_index != completed_blocks:
        raise RuntimeError("Recovered carry boundary does not match the checkpoint")
    return {
        "first_replayed_episode": first_episode,
        "last_replayed_episode": episode,
        "replayed_transitions_without_learning": transitions_needed,
        "max_state_abs_error": max_state_error,
        "max_reward_abs_error": max_reward_error,
        "episode_total_reward_abs_errors": episode_reward_errors,
        "hard_reset_count": env.hard_reset_count,
        "carry_block_index": env.carry_block_index,
        "policy_time_seconds": env.t,
        "native_time_seconds": float(env.obs.lib.get_time_c()),
        "reference_selection_count": reference._episode_index,
        "active_reference_member": int(reference._active_index),
        "reference_visit_counts": reference.visit_counts.tolist(),
        "scenario": getattr(env, "current_dynamics_scenario", None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "dr"), required=True)
    parser.add_argument("--checkpoint-episode", type=int, default=1500)
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Final total episode count, not additional episodes")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--name")
    parser.add_argument("--recovery-dir", type=Path,
                        default=ROOT / "pneu_rl/exp/lib3_hrr_dr_20261026/recovery_20261026_2311")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.threads < 1 or not 0 < args.checkpoint_episode < args.episodes:
        parser.error("Require positive threads and 0 < checkpoint < final episodes")
    torch.set_num_threads(args.threads)
    name = args.name or f'261026_lib3_pressurecal_type1_HRR{"_DR" if args.variant == "dr" else ""}_smooth_seed0'
    if Path(name).name != name:
        parser.error("name must be one model folder")
    folder = ROOT / "pneu_rl/models" / name
    cfg = yaml.safe_load((folder / "cfg.yaml").read_text())
    if cfg["episodes"] != args.episodes or cfg["model"]["horizon"] != 512:
        raise ValueError("Saved total episodes/horizon do not match requested recovery")
    checkpoint = folder / f"model_epi_{args.checkpoint_episode:04d}.pth"
    replay_file = folder / "buffer.pkl"
    with (folder / "infos.pkl").open("rb") as handle:
        infos = pickle.load(handle)
    if args.checkpoint_episode not in infos:
        raise RuntimeError("Checkpoint episode is missing from saved training information")
    if not args.dry_run and max(infos) != args.checkpoint_episode:
        raise RuntimeError("Archive and trim incomplete post-checkpoint infos before resuming")
    infos = {k: v for k, v in infos.items() if k <= args.checkpoint_episode}
    seed_everything(int(cfg["train_seed"]))
    env = make_training_components(cfg, scenario_log=folder / "dynamics_randomization.jsonl")
    model = SAC(env=env, **cfg["model"])
    model.set_logger(name)
    model.logger.infos = infos
    model.last_epi = args.checkpoint_episode
    model.total_steps = int(infos[args.checkpoint_episode]["steps"])
    if model.total_steps != args.checkpoint_episode * model.horizon:
        raise RuntimeError("Checkpoint step count does not match complete episodes")
    model.load_model(str(checkpoint), evaluate=False)
    model.is_model_loaded = True
    # Trusted, local experiment pickle. Read exactly this file; do not silently
    # select a backup or duplicate the multi-GB buffer on disk/in memory.
    with replay_file.open("rb") as handle:
        model.buffer.buffer = pickle.load(handle)
    expected_replay_length = min(model.total_steps, int(cfg["model"]["buffer_size"]))
    if len(model.buffer) != expected_replay_length:
        raise RuntimeError("Replay length is inconsistent with the checkpoint")
    actor_hash = policy_hash(model)
    restoration = restore_physical_rollout(
        env, model, cfg, args.checkpoint_episode, folder / "dynamics_randomization.jsonl")
    if actor_hash != policy_hash(model):
        raise RuntimeError("Environment restoration unexpectedly changed policy weights")
    # Explicit new stochastic training segment. The original stochastic streams
    # are unavailable; both comparison arms receive the same declared seed.
    seed_everything(int(cfg["train_seed"]))
    now = time.time()
    record = {
        "variant": args.variant, "model": name, "dry_run": args.dry_run,
        "started_unix": now, "checkpoint_episode": args.checkpoint_episode,
        "target_episode": args.episodes, "restored_policy_steps": model.total_steps,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "restored_actor_sha256": actor_hash, "replay_file": str(replay_file),
        "replay_bytes": replay_file.stat().st_size,
        "replay_entries": len(model.buffer),
        "replay_modified_unix": replay_file.stat().st_mtime,
        "physical_rollout_restoration": restoration,
        "training_rng_seed_for_recovery_segment": int(cfg["train_seed"]),
        "restored": ["actor", "critic", "target_critic", "actor_optimizer",
                     "critic_optimizer", "learned_log_alpha", "replay_buffer",
                     "physical_rollout", "reference_coverage_rng", "dynamics_rng"],
        "not_checkpointed": ["alpha_optimizer_moments", "policy_sampling_rng",
                             "replay_sampling_rng"],
        "bit_identical_uninterrupted_training": False,
        "torch_threads": args.threads,
        "completed": False,
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.dry_run:
        env.close()
        return

    report_path = args.recovery_dir / f"{args.variant}_resume.json"
    wall_path = folder / "wall_time.json"
    wall = json.loads(wall_path.read_text())
    original_start = wall["started_unix"]
    estimated_prior_active = max(0.0, replay_file.stat().st_mtime - original_start)
    wall.setdefault("recovery_segments", []).append(record)
    wall["completed"] = False
    wall["active_recovery_started_unix"] = now
    save_json(report_path, record)
    save_json(wall_path, wall)
    try:
        model.train(episode=args.episodes - args.checkpoint_episode)
        record["completed"] = max(model.logger.infos) == args.episodes
    finally:
        finished = time.time()
        record.update(finished_unix=finished, elapsed_seconds=finished - now,
                      final_episode=max(model.logger.infos),
                      final_policy_steps=model.total_steps)
        wall.update(finished_unix=finished,
                    elapsed_seconds=finished - original_start,
                    estimated_active_seconds=estimated_prior_active + finished - now,
                    elapsed_includes_interruption_gap=True,
                    completed=record["completed"], policy_steps=model.total_steps)
        save_json(report_path, record)
        save_json(wall_path, wall)
        env.close()


if __name__ == "__main__":
    main()
