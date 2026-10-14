#!/usr/bin/env python3
"""Deterministically evaluate lib4 screening models on every ND-bank member."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_rl" / "src",
    ROOT / "pneu_utils" / "src",
    SCRIPT_DIR,
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from train_lib4_screening import make_components, resolve_profile
from lib4_evaluation_protocol import load_protocol, summarize_validation
from pneu_rl.sac_with_loss import SAC


CHANNELS = (
    "ctrl_ch_pos",
    "ctrl_ch_neg",
    "ctrl_act_pos_in",
    "ctrl_act_pos_out",
    "ctrl_act_neg_in",
    "ctrl_act_neg_out",
)


def save_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_member(output_dir: Path, member: int, rows: list[dict[str, float]]) -> None:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    time = values("time") - values("time")[0]
    main_specs = (
        ("ch_pos", None, "Chamber Pos [kPa]", "red"),
        ("ch_neg", None, "Chamber Neg [kPa]", "blue"),
        ("act_pos", "ref_pos", "Act Pos [kPa]", "red"),
        ("act_neg", "ref_neg", "Act Neg [kPa]", "blue"),
        ("disp", "ref_disp", "Displacement [mm]", "green"),
    )
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), constrained_layout=True)
    for axis, (actual_key, ref_key, ylabel, color) in zip(axes, main_specs):
        if ref_key is not None:
            axis.plot(time, values(ref_key), color="black", linewidth=2.0, label="REF")
        axis.plot(time, values(actual_key), color=color, linewidth=1.8, label=actual_key.upper())
        axis.set_xlabel("Time [sec]")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.35)
        axis.legend(loc="best")
    fig.suptitle(f"lib4 deterministic validation — member {member:02d}")
    fig.savefig(output_dir / f"member_{member:02d}_main.png", dpi=160)
    plt.close(fig)

    colors = ("red", "blue", "tab:orange", "tab:purple", "tab:green", "tab:brown")
    fig, axes = plt.subplots(6, 1, figsize=(13, 14), constrained_layout=True)
    for axis, key, color in zip(axes, CHANNELS, colors):
        axis.plot(time, values(key), color=color, linewidth=1.6, label=key.upper())
        axis.set_xlabel("Time [sec]")
        axis.set_ylabel("Control")
        axis.set_ylim(0.79, 1.01)
        axis.grid(True, which="both", alpha=0.35)
        axis.legend(loc="best")
    fig.suptitle(f"lib4 six-control validation — member {member:02d}")
    fig.savefig(output_dir / f"member_{member:02d}_ctrl.png", dpi=160)
    plt.close(fig)


def evaluate_model(
    model_name: str,
    profile_path: Path,
    output_root: Path,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    model_dir = ROOT / "pneu_rl" / "models" / model_name
    with (model_dir / "cfg.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    profile, resolved_profile_path = resolve_profile(profile_path)
    cfg["reference"] = profile
    cfg["reference_profile_source"] = str(resolved_profile_path)

    env = make_components(cfg)
    model = SAC(env=env, **cfg["model"])
    model.load_model(str(model_dir / "model.pth"), evaluate=True)
    bank = env.goal.ref
    if not hasattr(bank, "reference") or not hasattr(bank, "visit_counts"):
        raise TypeError("lib4 evaluation requires a trajectory-bank reference")
    bank.visit_counts.fill(0)
    bank.selection_log.clear()
    bank.rng.seed(int(profile["kwargs"].get("seed", 0)))

    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_horizon = int(cfg["physical_horizon"])
    chunk = int(cfg["action_chunk_steps"])
    policy_horizon = physical_horizon // chunk
    carry_blocks = int(cfg["env"]["episode_carry_blocks"])
    member_count = int(bank.reference.shape[0])
    summaries: list[dict[str, float | int]] = []

    try:
        for _ in range(member_count):
            rows: list[dict[str, float]] = []
            member = -1
            policy_step_index = 0
            for block in range(carry_blocks):
                state, _ = env.reset()
                if block == 0:
                    member = int(bank._active_index)
                elif int(bank._active_index) != member:
                    raise AssertionError("reference member changed inside a carry sequence")
                for _ in range(policy_horizon):
                    with torch.no_grad():
                        action = np.asarray(model.evaluate_action(state), dtype=np.float64)
                    # Save a deterministic critic value alongside each reward
                    # so the same validation trace can be audited with the
                    # offline TD(lambda) discrepancy tool.  For K>1 the
                    # policy decision value is repeated over its physical
                    # sub-steps; K=1 is the exact one-step value trace.
                    value_estimate = model.estimate_value(state, action)
                    sub_actions = env.decode_action_chunk(action)
                    if len(sub_actions) != chunk:
                        raise AssertionError(
                            f"decoded {len(sub_actions)} sub-actions for K={chunk}"
                        )
                    for sub_action_index, sub_action in enumerate(sub_actions):
                        state, reward, _, _, info = env._step_single(sub_action)
                        obs = info["obs"]
                        raw = np.asarray(info["ctrl_input"], dtype=np.float64).reshape(-1, env.dim_act)[0]
                        rows.append(
                            {
                                "policy_step": int(policy_step_index),
                                "sub_action_index": int(sub_action_index),
                                "time": float(obs["curr_time"]),
                                "ch_pos": float(obs["sen_pos"]),
                                "ch_neg": float(obs["sen_neg"]),
                                "act_pos": float(obs["sen_act_pos"]),
                                "act_neg": float(obs["sen_act_neg"]),
                                "disp": float(obs["sen_env_displacement"]),
                                "ref_pos": float(obs["ref_act_pos"]),
                                "ref_neg": float(obs["ref_act_neg"]),
                                "ref_disp": float(obs["ref_displacement"]),
                                "ctrl_ch_pos": float(obs["ctrl_pos"]),
                                "ctrl_ch_neg": float(obs["ctrl_neg"]),
                                "ctrl_act_pos_in": float(obs["ctrl_act_pos_in"]),
                                "ctrl_act_pos_out": float(obs["ctrl_act_pos_out"]),
                                "ctrl_act_neg_in": float(obs["ctrl_act_neg_in"]),
                                "ctrl_act_neg_out": float(obs["ctrl_act_neg_out"]),
                                "raw_ctrl_ch_pos": float(raw[0]),
                                "raw_ctrl_ch_neg": float(raw[1]),
                                "raw_ctrl_act_pos_in": float(raw[2]),
                                "raw_ctrl_act_pos_out": float(raw[3]),
                                "raw_ctrl_act_neg_in": float(raw[4]),
                                "raw_ctrl_act_neg_out": float(raw[5]),
                                "reward": float(reward),
                                "value": float(value_estimate),
                            }
                        )
                    policy_step_index += 1
            if len(rows) != physical_horizon * carry_blocks:
                raise AssertionError(
                    f"member {member}: {len(rows)} actual steps != "
                    f"{physical_horizon * carry_blocks}"
                )
            member_dir = output_dir / f"member_{member:02d}"
            member_dir.mkdir(parents=True, exist_ok=True)
            save_rows(member_dir / "validation.csv", rows)
            plot_member(member_dir, member, rows)
            summaries.append(
                summarize_validation(
                    rows,
                    member=member,
                    action_chunk_steps=chunk,
                    protocol=protocol,
                )
            )
    finally:
        env.close()

    summaries.sort(key=lambda row: int(row["member"]))
    save_rows(output_dir / "members.csv", summaries)
    metric_keys = [key for key in summaries[0] if key not in {"member", "actual_steps"}]
    aggregate: dict[str, Any] = {
        "model": model_name,
        "trained_episodes": int(cfg["episodes"]),
        "checkpoint_logged_episodes": 0,
        "reference_profile": str(profile_path),
        "members": member_count,
        "actual_steps_per_member": physical_horizon * carry_blocks,
        "action_chunk_steps": chunk,
        "action_chunk_mode": str(getattr(env, "action_chunk_mode", "free")),
        "evaluation_protocol": str(protocol_path),
        "evaluation_protocol_sha256": protocol_sha256,
        "all_members_pass": bool(all(bool(row["gate_all"]) for row in summaries)),
    }
    diagnostics_path = model_dir / "train_diagnostics.csv"
    if diagnostics_path.is_file():
        with diagnostics_path.open(encoding="utf-8") as handle:
            aggregate["checkpoint_logged_episodes"] = sum(1 for _ in handle) - 1
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in summaries], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
        aggregate[f"{key}_min"] = float(np.min(values))
        aggregate[f"{key}_max"] = float(np.max(values))
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(aggregate, handle, sort_keys=False)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "pneu_rl/config/evaluation/lib4_p0_protocol.yaml",
    )
    args = parser.parse_args()
    output_root = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_sha256 = load_protocol(protocol_path)
    results = [
        evaluate_model(
            name,
            args.reference_profile,
            output_root,
            protocol,
            protocol_path,
            protocol_sha256,
        )
        for name in args.models
    ]
    with (output_root / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "evaluation_protocol": str(protocol_path),
                "evaluation_protocol_sha256": protocol_sha256,
                "models": results,
            },
            handle,
            sort_keys=False,
        )
    print(
        yaml.safe_dump(
            {
                "evaluation_protocol": str(protocol_path),
                "evaluation_protocol_sha256": protocol_sha256,
                "models": results,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
