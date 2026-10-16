"""Evaluate and visualize a saved lib3 screening model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from pneu_rl.plotting import plot_lib3_pressure_and_controls
from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import get_pkg_path


MODELS_DIR = Path(get_pkg_path("pneu_rl")) / "models"


def legacy_lib3_reference(seed: int) -> dict:
    """Build the historical lib3 RandomRef without changing model configs."""
    return {
        "kind": "legacy_random",
        "kwargs": {
            "seed": int(seed),
            "pos_min_off": 115,
            "pos_max_off": 140,
            "neg_min_off": 60,
            "neg_max_off": 80,
            "pos_max_ts": 5,
            "neg_max_ts": 5,
            "pos_max_amp": 10,
            "neg_max_amp": 5,
            "pos_max_per": 10,
            "neg_max_per": 10,
        },
    }


def build_model_env(
    model_name: str,
    reference_kind: str | None = None,
    ref_seed: int | None = None,
    checkpoint_episode: int | None = None,
):
    model_dir = MODELS_DIR / model_name
    with (model_dir / "cfg.yaml").open(encoding="utf-8") as handle:
        # Training configs contain tuple-valued action masks emitted by the
        # existing YAML serializer.  This is a local, trusted model config;
        # use FullLoader so the evaluator reads the exact training mask.
        cfg = yaml.full_load(handle)
    if int(cfg["lib"]) != 3:
        raise ValueError(f"{model_name} is not a lib3 model")

    if reference_kind == "legacy_random":
        cfg["reference"] = legacy_lib3_reference(
            int(cfg.get("ref_seed", 61098) if ref_seed is None else ref_seed)
        )

    from pneu_env.env3 import PneuEnv3
    from pneu_env.pred3 import PneuPred
    from pneu_env.sim3 import PneuSim

    reference = make_pressure_reference(reference_spec_from_model_config(cfg))
    obs = PneuSim(**cfg["obs"])
    pred = PneuPred(**cfg["pred"]) if cfg.get("pred") is not None else None
    env_cfg = dict(cfg["env"])
    env_cfg["verbose"] = False
    initialization = cfg.get("initialization") or {}
    if initialization.get("mode") == "pump_on_all_open":
        from pneu_env.env3_all_open import PneuEnv3AllOpen
        env = PneuEnv3AllOpen(
            obs=obs, ref=reference, pred=pred,
            precondition_seconds=initialization["precondition_seconds"], **env_cfg,
        )
    else:
        env = PneuEnv3(obs=obs, ref=reference, pred=pred, **env_cfg)
    if cfg.get("pid") is not None:
        env.set_pid(**cfg["pid"])

    model = SAC(env=env, **cfg["model"])
    model.set_logger(model_name)
    model_path = Path(model.logger.model_path)
    if checkpoint_episode is not None:
        model_path = model_dir / f"model_epi_{checkpoint_episode:04d}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model.load_model(path=str(model_path), evaluate=True)
    return cfg, env, model


def _row_from_info(env, info: dict, reward: float, step: int) -> dict:
    """Convert one *physical* simulator substep into an evaluation row."""

    obs = info["obs"]
    applied_a = np.asarray(info["ctrl_input"], dtype=np.float64).reshape(-1, 6)[0]
    scale_low = float(getattr(env.obs, "scale_low", 0.0))
    scale_high = float(getattr(env.obs, "scale_high", 1.0))
    applied_physical = scale_low + (scale_high - scale_low) * applied_a
    row = {
        "step": step,
        "time": float(obs["curr_time"]),
        "ref_pos": float(obs["ref_act_pos"]),
        "ref_neg": float(obs["ref_act_neg"]),
        # Chamber references are derived from the same two actuator
        # references used by training.  They are not independent random
        # channels: POS keeps +35 kPa reserve and NEG keeps -35 kPa.
        "chamber_ref_pos": float(np.clip(obs["ref_act_pos"] + 35.0, 101.325, 200.0)),
        "chamber_ref_neg": float(np.clip(obs["ref_act_neg"] - 35.0, 15.0, 101.325)),
        "act_pos": float(obs["sen_act_pos"]),
        "act_neg": float(obs["sen_act_neg"]),
        "chamber_pos": float(obs["sen_pos"]),
        "chamber_neg": float(obs["sen_neg"]),
        "reward": float(reward),
    }
    controller_target = np.asarray(
        info.get("chamber_target", [np.nan, np.nan]), dtype=np.float64
    ).reshape(-1)
    row["controller_chamber_ref_pos"] = float(controller_target[0])
    row["controller_chamber_ref_neg"] = float(controller_target[1])
    for index in range(6):
        row[f"ctrl{index + 1}_a"] = float(applied_a[index])
        # The primary ctrl column is the command actually sent to the
        # physical valve model.  Keep normalized a separately to avoid
        # confusing plots with the policy-space action.
        row[f"ctrl{index + 1}"] = float(applied_physical[index])
    return row


def run_episode(env, model, steps: int, state=None) -> pd.DataFrame:
    if state is None:
        state, _ = env.reset()
    rows = []
    physical_step = 0
    for _ in range(steps):
        action = model.predict(state)
        state, reward, _, _, info = env.step(action)
        substep_infos = info.get("chunk_substep_infos")
        substep_rewards = info.get("chunk_substep_rewards")
        if substep_infos is None or substep_rewards is None:
            substep_infos = [info]
            substep_rewards = [reward]
        for sub_info, sub_reward in zip(substep_infos, substep_rewards):
            rows.append(_row_from_info(env, sub_info, float(sub_reward), physical_step))
            physical_step += 1
    return pd.DataFrame(rows)


def reset_to_reference_member(env, member_index: int):
    """Hard-reset the plant onto one exact optimized-bank member.

    Coverage sampling is intentionally randomized within the least-visited
    set.  That is desirable in training but unsuitable for a paired model
    comparison, where every controller must see every member exactly once.
    Bias the next coverage selection toward the requested member, then verify
    that the reset selected it.
    """
    bank = getattr(getattr(env, "goal", None), "ref", None)
    trajectories = getattr(bank, "trajectories", None)
    visit_counts = getattr(bank, "visit_counts", None)
    if trajectories is None or visit_counts is None:
        raise ValueError("exact reference-member evaluation requires a trajectory bank")
    if getattr(bank, "sample_mode", None) != "coverage":
        raise ValueError("exact reference-member evaluation requires sample_mode=coverage")
    if member_index < 0 or member_index >= len(trajectories):
        raise IndexError(f"reference member {member_index} is outside [0, {len(trajectories)})")

    visit_counts[:] = 1
    visit_counts[member_index] = 0
    env.request_hard_reset()
    state, info = env.reset()
    if int(bank._active_index) != int(member_index):
        raise RuntimeError(
            f"requested reference member {member_index}, selected {bank._active_index}"
        )
    return state, info


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dt = float(np.median(np.diff(result["time"])))
    for name in ("ref_pos", "ref_neg", "act_pos", "act_neg"):
        result[f"rate_{name}"] = np.gradient(result[name].to_numpy(), dt)
    return result


def lagged_direction_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    dt: float,
    rate_threshold: float,
    max_lag_steps: int = 15,
) -> dict[str, float]:
    """Score whether actual pressure moves with a changing reference.

    A five-sample moving average suppresses one-step valve chatter.  The score
    then searches only 0--300 ms, covering the configured 100 ms delay without
    allowing an arbitrarily late trajectory to look correct.
    """

    kernel = np.full(5, 0.2, dtype=np.float64)
    ref_smooth = np.convolve(reference, kernel, mode="same")
    act_smooth = np.convolve(actual, kernel, mode="same")
    ref_rate = np.gradient(ref_smooth, dt)
    act_rate = np.gradient(act_smooth, dt)
    # Exclude startup and convolution edges.
    start = max(int(round(1.0 / dt)), 3)
    stop = len(reference) - 3
    best = {
        "agreement": float("nan"),
        "correlation": float("nan"),
        "lag_steps": 0,
        "evaluated_fraction": 0.0,
    }
    best_key = (-np.inf, -np.inf)
    for lag in range(max(int(max_lag_steps), 0) + 1):
        end = stop - lag
        if end <= start:
            continue
        ref_slice = ref_rate[start:end]
        act_slice = act_rate[start + lag:stop]
        mask = np.abs(ref_slice) >= float(rate_threshold)
        if not np.any(mask):
            continue
        ref_eval = ref_slice[mask]
        act_eval = act_slice[mask]
        agreement = float(np.mean(np.sign(ref_eval) == np.sign(act_eval)))
        if np.std(ref_eval) > 1e-9 and np.std(act_eval) > 1e-9:
            correlation = float(np.corrcoef(ref_eval, act_eval)[0, 1])
        else:
            correlation = 0.0
        key = (agreement, correlation)
        if key > best_key:
            best_key = key
            best = {
                "agreement": agreement,
                "correlation": correlation,
                "lag_steps": int(lag),
                "evaluated_fraction": float(np.mean(mask)),
            }
    best["lag_sec"] = float(best["lag_steps"] * dt)
    return best


def metrics(df: pd.DataFrame, tolerance: float) -> dict:
    data = add_rates(df)
    dt = float(np.median(np.diff(data["time"])))
    pos_error = data["act_pos"] - data["ref_pos"]
    neg_error = data["act_neg"] - data["ref_neg"]
    chamber_pos_error = data["chamber_pos"] - data["chamber_ref_pos"]
    chamber_neg_error = data["chamber_neg"] - data["chamber_ref_neg"]
    pos_direction = lagged_direction_metrics(
        data["ref_pos"].to_numpy(),
        data["act_pos"].to_numpy(),
        dt,
        rate_threshold=1.0,
    )
    neg_direction = lagged_direction_metrics(
        data["ref_neg"].to_numpy(),
        data["act_neg"].to_numpy(),
        dt,
        rate_threshold=0.5,
    )
    pos_atmosphere_rmse = float(
        np.sqrt(np.mean((data["ref_pos"].to_numpy() - 101.325) ** 2))
    )
    neg_atmosphere_rmse = float(
        np.sqrt(np.mean((data["ref_neg"].to_numpy() - 101.325) ** 2))
    )
    pos_rmse = float(np.sqrt(np.mean(pos_error**2)))
    neg_rmse = float(np.sqrt(np.mean(neg_error**2)))
    pos_reserve = data["chamber_pos"].to_numpy() - data["act_pos"].to_numpy()
    neg_reserve = data["act_neg"].to_numpy() - data["chamber_neg"].to_numpy()
    control_delta = np.diff(
        data[[f"ctrl{index}" for index in range(1, 7)]].to_numpy(),
        axis=0,
    )
    # The base training reward only measures absolute tracking error.  Report
    # a separate plateau-motion metric so a policy with similar MAE but higher
    # pressure ripple is not called equivalent.  The first second is excluded
    # to avoid reset transients; rate thresholds match the direction metrics.
    plateau_start = min(max(int(round(1.0 / dt)), 1), len(data) - 1)
    pos_plateau = np.zeros(len(data), dtype=bool)
    neg_plateau = np.zeros(len(data), dtype=bool)
    pos_plateau[plateau_start:] = (
        np.abs(data["rate_ref_pos"].to_numpy()[plateau_start:]) < 1.0
    )
    neg_plateau[plateau_start:] = (
        np.abs(data["rate_ref_neg"].to_numpy()[plateau_start:]) < 0.5
    )

    def masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
        selected = np.asarray(values, dtype=np.float64)[mask]
        return float(np.sqrt(np.mean(selected**2))) if selected.size else float("nan")

    plateau_mask = pos_plateau | neg_plateau
    plateau_action_delta = control_delta[plateau_mask[1:]]
    return {
        "steps": int(len(data)),
        "duration_sec": float(data["time"].iloc[-1]),
        "episode_reward": float(data["reward"].sum()),
        "pos_rmse_kpa": pos_rmse,
        "neg_rmse_kpa": neg_rmse,
        "pos_mae_kpa": float(np.mean(np.abs(pos_error))),
        "neg_mae_kpa": float(np.mean(np.abs(neg_error))),
        "chamber_pos_rmse_kpa": float(np.sqrt(np.mean(chamber_pos_error**2))),
        "chamber_neg_rmse_kpa": float(np.sqrt(np.mean(chamber_neg_error**2))),
        "pos_tolerance_fraction": float(np.mean(np.abs(pos_error) <= tolerance)),
        "neg_tolerance_fraction": float(np.mean(np.abs(neg_error) <= tolerance)),
        "ref_pos_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_ref_pos"]))),
        "ref_neg_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_ref_neg"]))),
        "act_pos_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_act_pos"]))),
        "act_neg_rate_abs_max_kpa_s": float(np.max(np.abs(data["rate_act_neg"]))),
        "pos_atmosphere_hold_rmse_kpa": pos_atmosphere_rmse,
        "neg_atmosphere_hold_rmse_kpa": neg_atmosphere_rmse,
        "pos_improvement_over_atmosphere": float(
            1.0 - pos_rmse / max(pos_atmosphere_rmse, 1e-9)
        ),
        "neg_improvement_over_atmosphere": float(
            1.0 - neg_rmse / max(neg_atmosphere_rmse, 1e-9)
        ),
        "pos_actual_span_kpa": float(data["act_pos"].max() - data["act_pos"].min()),
        "neg_actual_span_kpa": float(data["act_neg"].max() - data["act_neg"].min()),
        "pos_direction_agreement": pos_direction["agreement"],
        "neg_direction_agreement": neg_direction["agreement"],
        "pos_direction_correlation": pos_direction["correlation"],
        "neg_direction_correlation": neg_direction["correlation"],
        "pos_best_lag_sec": pos_direction["lag_sec"],
        "neg_best_lag_sec": neg_direction["lag_sec"],
        "pos_direction_evaluated_fraction": pos_direction["evaluated_fraction"],
        "neg_direction_evaluated_fraction": neg_direction["evaluated_fraction"],
        "positive_reserve_mean_kpa": float(np.mean(pos_reserve)),
        "negative_reserve_mean_kpa": float(np.mean(neg_reserve)),
        "positive_reserve_ge_10kpa_fraction": float(np.mean(pos_reserve >= 10.0)),
        "negative_reserve_ge_10kpa_fraction": float(np.mean(neg_reserve >= 10.0)),
        "chamber_pos_min_kpa": float(data["chamber_pos"].min()),
        "chamber_pos_max_kpa": float(data["chamber_pos"].max()),
        "chamber_neg_min_kpa": float(data["chamber_neg"].min()),
        "chamber_neg_max_kpa": float(data["chamber_neg"].max()),
        "physical_action_delta_abs_mean": float(np.mean(np.abs(control_delta))),
        "physical_action_delta_abs_max": float(np.max(np.abs(control_delta))),
        "plateau_pos_pressure_rate_rms_kpa_s": masked_rms(
            data["rate_act_pos"].to_numpy(), pos_plateau
        ),
        "plateau_neg_pressure_rate_rms_kpa_s": masked_rms(
            data["rate_act_neg"].to_numpy(), neg_plateau
        ),
        "plateau_action_delta_rms": (
            float(np.sqrt(np.mean(plateau_action_delta**2)))
            if plateau_action_delta.size
            else float("nan")
        ),
        "ctrl1_mean": float(data["ctrl1"].mean()),
        "ctrl2_mean": float(data["ctrl2"].mean()),
        "ctrl3_mean": float(data["ctrl3"].mean()),
        "ctrl4_mean": float(data["ctrl4"].mean()),
        "ctrl5_mean": float(data["ctrl5"].mean()),
        "ctrl6_mean": float(data["ctrl6"].mean()),
    }


def plot_episode(
    df: pd.DataFrame, prefix: Path
) -> tuple[Path | None, Path | None]:
    return plot_lib3_pressure_and_controls(df, prefix)


def evaluate(
    model_name: str,
    episodes: int,
    steps: int,
    tolerance: float,
    output_root: Path,
    reference_kind: str | None = None,
    ref_seed: int | None = None,
    checkpoint_episode: int | None = None,
    all_reference_members: bool = False,
    make_plots: bool = True,
) -> None:
    cfg, env, model = build_model_env(
        model_name,
        reference_kind,
        ref_seed,
        checkpoint_episode,
    )
    output_name = (
        model_name
        if checkpoint_episode is None
        else f"{model_name}_epi{checkpoint_episode:04d}"
    )
    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = getattr(getattr(env, "goal", None), "ref", None)
    member_indices = (
        list(range(len(bank.trajectories)))
        if all_reference_members
        else [None] * episodes
    )
    episode_metrics = []
    try:
        for episode, member_index in enumerate(member_indices, start=1):
            state = None
            stem = f"episode_{episode:02d}"
            if member_index is not None:
                state, _ = reset_to_reference_member(env, member_index)
                stem = f"member_{member_index:02d}"
            df = run_episode(env, model, steps, state=state)
            df.to_csv(output_dir / f"{stem}.csv", index=False)
            if make_plots:
                plot_episode(df, output_dir / stem)
            row = {"episode": episode, **metrics(df, tolerance)}
            if member_index is not None:
                row["reference_member"] = int(member_index)
            episode_metrics.append(row)
    finally:
        env.close()

    summary = {
        "model": model_name,
        "checkpoint_episode": checkpoint_episode,
        "variant": cfg["variant"],
        "reference": cfg["reference"],
        "pid_enabled": bool(cfg.get("pid") is not None),
        "episodes": episode_metrics,
        "mean": {
            key: float(np.mean([row[key] for row in episode_metrics]))
            for key in episode_metrics[0]
            if key not in {"episode", "reference_member"}
        },
        "std": {
            key: float(np.std([row[key] for row in episode_metrics]))
            for key in episode_metrics[0]
            if key not in {"episode", "reference_member"}
        },
    }
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(yaml.safe_dump(summary["mean"], sort_keys=False))
    print(f"saved: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=Path("pneu_rl/exp/lib3_screening"))
    parser.add_argument(
        "--checkpoint-episode",
        type=int,
        default=None,
        help="evaluate model_epi_NNNN.pth instead of the rolling model.pth",
    )
    parser.add_argument(
        "--reference-kind",
        choices=("legacy_random",),
        default=None,
        help="override the saved reference for evaluation without editing cfg.yaml",
    )
    parser.add_argument(
        "--ref-seed",
        type=int,
        default=None,
        help="reference seed used with --reference-kind",
    )
    parser.add_argument(
        "--all-reference-members",
        action="store_true",
        help="hard-reset and evaluate every optimized-bank member exactly once",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip PNG generation (useful for full-bank validation)",
    )
    args = parser.parse_args()
    if args.checkpoint_episode is not None and args.checkpoint_episode <= 0:
        parser.error("--checkpoint-episode must be positive")
    for model_name in args.model:
        evaluate(
            model_name,
            args.episodes,
            args.steps,
            args.tolerance,
            args.output_root,
            args.reference_kind,
            args.ref_seed,
            args.checkpoint_episode,
            args.all_reference_members,
            not args.no_plot,
        )


if __name__ == "__main__":
    main()
