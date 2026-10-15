"""Evaluate a saved screening model on its exact training configuration.

Unlike the legacy interactive quick-viz scripts, this evaluator loads the
model's own cfg.yaml.  Therefore the reference distribution, action scaling,
MPObs setting, PID setting, and initial state match training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import yaml

from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_rl.sac_with_loss import SAC
from pneu_rl.exp_paths import ensure_model_exp_dir
from pneu_utils.utils import get_pkg_path


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "pneu_rl" / "models"
DEFAULT_CAPACITY_PROFILE = (
    ROOT / "pneu_rl" / "config" / "reference" / "lib2_feasible.yaml"
)
EVAL_STEPS = 2048
EVAL_DURATION_SEC = EVAL_STEPS / 50.0
FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
TIMES = fm.FontProperties(fname=FONT_PATH)
RED = "red"
BLUE = "blue"
BLACK = "black"


def reference_sha256(reference_spec: dict) -> str:
    """Return a content hash independent of YAML key insertion order."""
    canonical = yaml.safe_dump(reference_spec, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_optimized_bank_member(profile: dict, member: int) -> tuple[dict, int]:
    """Return one bank member as a standalone reference without changing its design.

    The full profile remains the identity used for training/evaluation hash
    matching.  The returned one-member profile merely makes per-trajectory
    performance auditable instead of letting a deterministic first member
    stand in for all ten training trajectories.
    """
    if profile.get("kind") != "optimized_trajectory_bank":
        raise ValueError("--bank-member requires an optimized_trajectory_bank profile")
    kwargs = dict(profile.get("kwargs", {}))
    trajectories = kwargs.get("trajectories")
    if trajectories is None:
        npz_value = kwargs.get("trajectory_npz")
        if npz_value is None:
            raise ValueError("optimized bank has neither inline trajectories nor trajectory_npz")
        npz_path = Path(npz_value)
        if not npz_path.is_absolute():
            npz_path = ROOT / npz_path
        with np.load(npz_path) as data:
            times = np.asarray(data["time"], dtype=np.float64)
            pressure = np.asarray(data["pressure"], dtype=np.float64)
        if pressure.ndim == 2:
            pressure = pressure[None, ...]
        if times.ndim == 1:
            times = np.repeat(times[None, :], pressure.shape[0], axis=0)
        trajectories = [
            {"id": index, "time": times[index].tolist(), "pressure": pressure[index].tolist()}
            for index in range(pressure.shape[0])
        ]
    if member < 0 or member >= len(trajectories):
        raise ValueError(f"--bank-member must be in [0, {len(trajectories) - 1}]")
    selected = trajectories[member]
    selected_id = int(selected.get("id", member))
    return {
        "kind": "optimized_trajectory_bank",
        "kwargs": {
            "trajectories": [selected],
            "seed": int(kwargs.get("seed", 61098)),
            "sample_mode": "random",
            "repeat_after_end": bool(kwargs.get("repeat_after_end", False)),
        },
        "design": dict(profile.get("design", {})),
    }, selected_id


def load_model_env(model_name: str, reference_spec=None, init_pos=None, init_neg=None):
    model_dir = MODELS_DIR / model_name
    with (model_dir / "cfg.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if int(cfg["lib"]) != 2:
        raise ValueError("evaluate_screening.py currently supports --lib 2 models only")

    from pneu_env.env import PneuEnv
    from pneu_env.pred2 import PneuPred
    from pneu_env.sim2 import PneuSim

    obs_cfg = dict(cfg["obs"])
    if init_pos is not None:
        obs_cfg["init_pos_press"] = float(init_pos)
    if init_neg is not None:
        obs_cfg["init_neg_press"] = float(init_neg)
    obs = PneuSim(**obs_cfg)
    pred_cfg = dict(cfg["pred"]) if cfg.get("pred") is not None else None
    pred = PneuPred(**pred_cfg) if pred_cfg is not None else None
    env_cfg = dict(cfg["env"])
    env_cfg["verbose"] = False
    active_reference = reference_spec or reference_spec_from_model_config(cfg)
    env = PneuEnv(
        obs=obs,
        pred=pred,
        ref=make_pressure_reference(active_reference),
        **env_cfg,
    )
    if cfg.get("pid") is not None:
        env.set_pid(**cfg["pid"])

    model = SAC(env=env, **cfg["model"])
    model.set_logger(model_name)
    model.load_model(path=model.logger.model_path, evaluate=True)
    model.reset_action_state()
    return cfg, env, model, active_reference


def sign_change_per_sec(values: np.ndarray, time_values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    delta = np.diff(values)
    signs = np.sign(delta)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    duration = max(float(time_values[-1] - time_values[0]), 1e-9)
    return float(np.mean(signs[1:] != signs[:-1]) * len(signs) / duration)


def slew_rate(values: np.ndarray, time_values: np.ndarray) -> np.ndarray:
    """Return |dx/dt| while ignoring duplicate simulator timestamps."""
    delta_t = np.diff(time_values)
    result = np.zeros_like(delta_t, dtype=np.float64)
    valid = delta_t > 1e-9
    result[valid] = np.abs(np.diff(values)[valid]) / delta_t[valid]
    return result


def max_or_zero(values: np.ndarray) -> float:
    """Return a numeric maximum for normal and one-step smoke rollouts."""
    return float(np.max(values)) if values.size else 0.0


def mean_abs_delta(values: np.ndarray) -> float:
    delta = np.diff(values)
    return float(np.mean(np.abs(delta))) if delta.size else 0.0


def directional_rate_utilization(
    values: np.ndarray,
    time_values: np.ndarray,
    rate_up: float,
    rate_down: float,
) -> np.ndarray:
    """Return requested signed slew divided by the matching plant limit."""
    delta_t = np.diff(time_values)
    delta = np.diff(values)
    valid = delta_t > 1e-9
    signed_rate = np.zeros_like(delta, dtype=np.float64)
    signed_rate[valid] = delta[valid] / delta_t[valid]
    limit = np.where(signed_rate >= 0.0, rate_up, rate_down)
    return np.abs(signed_rate) / np.maximum(limit, 1e-9)


def evaluate(
    model_name: str,
    steps: int,
    tolerance: float,
    reference_spec=None,
    output_tag=None,
    init_pos=None,
    init_neg=None,
    reference_identity=None,
    bank_member_index=None,
    bank_member_id=None,
):
    cfg, env, model, active_reference = load_model_env(
        model_name, reference_spec, init_pos=init_pos, init_neg=init_neg
    )
    training_reference = reference_spec_from_model_config(cfg)
    training_reference_hash = reference_sha256(training_reference)
    reference_identity = reference_identity or active_reference
    evaluation_reference_hash = reference_sha256(reference_identity)
    state = env.reset()[0]
    rows = []

    for step in range(int(steps)):
        action = model.evaluate_action(state)
        state, reward, _, _, info = env.step(action)
        obs = info["obs"]
        reward_info = info["reward"]
        valve_debug = env.obs.get_valve_debug()
        mean_flow = env.obs.get_mean_mass_flowrate()
        rows.append(
            dict(
                step=step,
                time=float(obs["curr_time"]),
                ref_pos=float(obs["ref_pos"]),
                ref_neg=float(obs["ref_neg"]),
                sen_pos=float(obs["sen_pos"]),
                sen_neg=float(obs["sen_neg"]),
                ctrl_pos=float(obs["ctrl_pos"]),
                ctrl_neg=float(obs["ctrl_neg"]),
                reward=float(reward),
                pos_error=float(obs["ref_pos"] - obs["sen_pos"]),
                neg_error=float(obs["ref_neg"] - obs["sen_neg"]),
                pos_curr_reward=float(reward_info["pos_curr_reward"]),
                neg_curr_reward=float(reward_info["neg_curr_reward"]),
                pos_u_eff=float(valve_debug["pos_u_eff"]),
                neg_u_eff=float(valve_debug["neg_u_eff"]),
                pos_q_static_lpm=float(valve_debug["pos_q_static_lpm"]),
                neg_q_static_lpm=float(valve_debug["neg_q_static_lpm"]),
                pos_q_pred_lpm=float(valve_debug["pos_q_pred_lpm"]),
                neg_q_pred_lpm=float(valve_debug["neg_q_pred_lpm"]),
                pos_mdot=float(valve_debug["pos_mdot"]),
                neg_mdot=float(valve_debug["neg_mdot"]),
                mean_pump_in_mdot=float(mean_flow["pump_in"]),
                mean_pump_out_mdot=float(mean_flow["pump_out"]),
                mean_valve_pos_mdot=float(mean_flow["valve_pos"]),
                mean_valve_neg_mdot=float(mean_flow["valve_neg"]),
            )
        )
    env.close()

    eval_name = f"screen_eval_{output_tag}_{steps}step" if output_tag else f"screen_eval_{steps}step"
    output_dir = ensure_model_exp_dir(model_name, eval_name)
    csv_path = output_dir / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    data = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
    # Prefer limits declared by the exact reference under evaluation.  The
    # old fallback profile is only valid when no reference-specific capacity
    # metadata exists; otherwise it can incorrectly label a simulator-backed
    # profile as infeasible.
    capacity_profile = active_reference if isinstance(active_reference, dict) else None
    capacity_design = (capacity_profile or {}).get("design", {})
    has_reference_rate_filter = "rate_filter_kpa_s" in capacity_design
    if not has_reference_rate_filter and DEFAULT_CAPACITY_PROFILE.is_file():
        with DEFAULT_CAPACITY_PROFILE.open(encoding="utf-8") as handle:
            capacity_profile = yaml.safe_load(handle)
            capacity_design = (capacity_profile or {}).get("design", {})
    capacity_kwargs = (capacity_profile or {}).get("kwargs", {})
    rate_filter = capacity_design.get("rate_filter_kpa_s")
    if isinstance(rate_filter, (list, tuple)) and len(rate_filter) >= 2:
        # Optimized profiles store symmetric magnitude filters because their
        # pressure trajectory was already generated by the full simulator.
        capacity_kwargs = dict(capacity_kwargs)
        capacity_kwargs.update(
            {
                "pos_rate_up": float(rate_filter[0]),
                "pos_rate_down": float(rate_filter[0]),
                "neg_rate_up": float(rate_filter[1]),
                "neg_rate_down": float(rate_filter[1]),
            }
        )
    rate_ignore_initial = float(capacity_design.get("rate_filter_ignore_initial_sec", 0.0))
    rate_keep = data["time"] >= rate_ignore_initial
    rate_filter_evaluated = bool(np.any(rate_keep) and np.sum(rate_keep) >= 2)
    if rate_filter_evaluated:
        pos_rate_util = directional_rate_utilization(
            data["ref_pos"][rate_keep],
            data["time"][rate_keep],
            float(capacity_kwargs.get("pos_rate_up", np.inf)),
            float(capacity_kwargs.get("pos_rate_down", np.inf)),
        )
        neg_rate_util = directional_rate_utilization(
            data["ref_neg"][rate_keep],
            data["time"][rate_keep],
            float(capacity_kwargs.get("neg_rate_up", np.inf)),
            float(capacity_kwargs.get("neg_rate_down", np.inf)),
        )
        joint_rate_util = np.maximum(pos_rate_util, neg_rate_util)
        rate_metrics = {
            "pos_ref_rate_utilization_p95": float(np.quantile(pos_rate_util, 0.95)),
            "neg_ref_rate_utilization_p95": float(np.quantile(neg_rate_util, 0.95)),
            "joint_ref_rate_utilization_max": float(np.max(joint_rate_util)),
            "joint_ref_rate_violation_ratio": float(np.mean(joint_rate_util > 1.0)),
        }
    else:
        rate_metrics = {
            "pos_ref_rate_utilization_p95": None,
            "neg_ref_rate_utilization_p95": None,
            "joint_ref_rate_utilization_max": None,
            "joint_ref_rate_violation_ratio": None,
        }
    metrics = dict(
        model=model_name,
        lib=int(cfg["lib"]),
        variant=cfg["variant"],
        control_mode=cfg.get("control_mode", "rl_plus_iterm" if cfg.get("pid") is not None else "pure_rl"),
        training_pid_enabled=cfg.get("pid") is not None,
        evaluation_pid_enabled=cfg.get("pid") is not None,
        iterm_enabled=cfg.get("pid") is not None or "iterm" in str(cfg.get("variant", "")),
        episode_carry_state=bool(cfg.get("env", {}).get("episode_carry_state", False)),
        action_scale_low=float(cfg.get("obs", {}).get("scale_low", 0.8)),
        action_scale_high=float(cfg.get("obs", {}).get("scale_high", 1.0)),
        evaluation_init_pos=float(init_pos)
        if init_pos is not None
        else float(cfg.get("obs", {}).get("init_pos_press", 101.325)),
        evaluation_init_neg=float(init_neg)
        if init_neg is not None
        else float(cfg.get("obs", {}).get("init_neg_press", 101.325)),
        reference_config=active_reference,
        reference_identity_config=reference_identity,
        reference_match_scope="bank_member" if bank_member_index is not None else "exact",
        reference_bank_member_index=bank_member_index,
        reference_bank_member_id=bank_member_id,
        training_reference_sha256=training_reference_hash,
        evaluation_reference_sha256=evaluation_reference_hash,
        reference_content_match=training_reference_hash == evaluation_reference_hash,
        steps=int(len(rows)),
        evaluation_duration_target_sec=EVAL_DURATION_SEC,
        duration_sec=float(data["time"][-1] - data["time"][0]),
        reward_total=float(np.sum(data["reward"])),
        reward_per_step=float(np.mean(data["reward"])),
        pos_rmse=float(np.sqrt(np.mean(data["pos_error"] ** 2))),
        neg_rmse=float(np.sqrt(np.mean(data["neg_error"] ** 2))),
        pos_mae=float(np.mean(np.abs(data["pos_error"]))),
        neg_mae=float(np.mean(np.abs(data["neg_error"]))),
        pos_bias=float(np.mean(data["pos_error"])),
        neg_bias=float(np.mean(data["neg_error"])),
        pos_abs_error_p95=float(np.quantile(np.abs(data["pos_error"]), 0.95)),
        neg_abs_error_p95=float(np.quantile(np.abs(data["neg_error"]), 0.95)),
        pos_success_ratio=float(np.mean(np.abs(data["pos_error"]) <= tolerance)),
        neg_success_ratio=float(np.mean(np.abs(data["neg_error"]) <= tolerance)),
        pos_ref_slew_max=max_or_zero(slew_rate(data["ref_pos"], data["time"])),
        neg_ref_slew_max=max_or_zero(slew_rate(data["ref_neg"], data["time"])),
        pos_pressure_slew_max=max_or_zero(slew_rate(data["sen_pos"], data["time"])),
        neg_pressure_slew_max=max_or_zero(slew_rate(data["sen_neg"], data["time"])),
        feasibility_capacity_profile=(
            "reference_profile.design.rate_filter_kpa_s"
            if has_reference_rate_filter
            else str(DEFAULT_CAPACITY_PROFILE) if capacity_profile is not None else None
        ),
        rate_filter_ignore_initial_sec=rate_ignore_initial,
        rate_filter_evaluated=rate_filter_evaluated,
        **rate_metrics,
        pos_ctrl_delta=mean_abs_delta(data["ctrl_pos"]),
        neg_ctrl_delta=mean_abs_delta(data["ctrl_neg"]),
        pos_ctrl_low_fraction=float(np.mean(data["ctrl_pos"] <= float(cfg.get("obs", {}).get("scale_low", 0.8)) + 1e-6)),
        neg_ctrl_low_fraction=float(np.mean(data["ctrl_neg"] <= float(cfg.get("obs", {}).get("scale_low", 0.8)) + 1e-6)),
        pos_ctrl_high_fraction=float(np.mean(data["ctrl_pos"] >= float(cfg.get("obs", {}).get("scale_high", 1.0)) - 1e-6)),
        neg_ctrl_high_fraction=float(np.mean(data["ctrl_neg"] >= float(cfg.get("obs", {}).get("scale_high", 1.0)) - 1e-6)),
        pos_ctrl_mean_headroom_low=float(np.mean(data["ctrl_pos"] - float(cfg.get("obs", {}).get("scale_low", 0.8)))),
        neg_ctrl_mean_headroom_low=float(np.mean(data["ctrl_neg"] - float(cfg.get("obs", {}).get("scale_low", 0.8)))),
        pos_ctrl_mean_headroom_high=float(np.mean(float(cfg.get("obs", {}).get("scale_high", 1.0)) - data["ctrl_pos"])),
        neg_ctrl_mean_headroom_high=float(np.mean(float(cfg.get("obs", {}).get("scale_high", 1.0)) - data["ctrl_neg"])),
        pos_pressure_sign_change_per_sec=sign_change_per_sec(data["sen_pos"], data["time"]),
        neg_pressure_sign_change_per_sec=sign_change_per_sec(data["sen_neg"], data["time"]),
        pos_u_eff_mean=float(np.mean(data["pos_u_eff"])),
        neg_u_eff_mean=float(np.mean(data["neg_u_eff"])),
        pos_u_eff_std=float(np.std(data["pos_u_eff"])),
        neg_u_eff_std=float(np.std(data["neg_u_eff"])),
        pos_q_static_mean_lpm=float(np.mean(data["pos_q_static_lpm"])),
        neg_q_static_mean_lpm=float(np.mean(data["neg_q_static_lpm"])),
        pos_q_pred_mean_lpm=float(np.mean(data["pos_q_pred_lpm"])),
        neg_q_pred_mean_lpm=float(np.mean(data["neg_q_pred_lpm"])),
        pos_mdot_mean=float(np.mean(data["pos_mdot"])),
        neg_mdot_mean=float(np.mean(data["neg_mdot"])),
        pump_in_mdot_mean=float(np.mean(data["mean_pump_in_mdot"])),
        pump_out_mdot_mean=float(np.mean(data["mean_pump_out_mdot"])),
        valve_pos_mdot_mean=float(np.mean(data["mean_valve_pos_mdot"])),
        valve_neg_mdot_mean=float(np.mean(data["mean_valve_neg_mdot"])),
    )
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics, handle, sort_keys=False)

    # Keep POS and NEG pressure on separate axes.  Their physical ranges are
    # very different (roughly 120--220 kPa vs. 10--30 kPa), so overlaying
    # them on one axis hides NEG oscillation and makes the screen plot
    # misleading.
    fig, axes = plt.subplots(6, 1, figsize=(13, 16), sharex=True)
    title = f"{model_name} | {cfg['variant']} | PID/I-term: {metrics['training_pid_enabled']}"
    fig.suptitle(title, fontproperties=TIMES, fontsize=17, fontweight="bold")

    axes[0].plot(data["time"], data["ref_pos"], "--", color=BLACK, linewidth=2, label="REF POS")
    axes[0].plot(data["time"], data["sen_pos"], color=RED, linewidth=2, label="POS")
    axes[0].set_ylabel("POS pressure [kPa]")
    axes[0].legend(fontsize=8)

    axes[1].plot(data["time"], data["ref_neg"], "--", color=BLACK, linewidth=2, label="REF NEG")
    axes[1].plot(data["time"], data["sen_neg"], color=BLUE, linewidth=2, label="NEG")
    axes[1].set_ylabel("NEG pressure [kPa]")
    axes[1].legend(fontsize=8)

    axes[2].plot(data["time"], data["pos_error"], color=RED, linewidth=2, label="POS error")
    axes[2].plot(data["time"], data["neg_error"], color=BLUE, linewidth=2, label="NEG error")
    axes[2].axhline(tolerance, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axhline(-tolerance, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Tracking error [kPa]")
    axes[2].legend(fontsize=8)

    axes[3].plot(data["time"], data["ctrl_pos"], color=RED, linewidth=2, label="POS")
    axes[3].plot(data["time"], data["ctrl_neg"], color=BLUE, linewidth=2, label="NEG")
    axes[3].set_ylabel("Control")
    axes[3].set_ylim(0.8, 1.0)
    axes[3].set_yticks(np.arange(0.8, 1.01, 0.05))
    axes[3].legend(fontsize=8)

    axes[4].plot(data["time"], data["pos_q_static_lpm"], color=RED, linestyle="--", linewidth=2, label="POS q static")
    axes[4].plot(data["time"], data["pos_q_pred_lpm"], color=RED, linewidth=2, label="POS q predicted")
    axes[4].plot(data["time"], data["neg_q_static_lpm"], color=BLUE, linestyle="--", linewidth=2, label="NEG q static")
    axes[4].plot(data["time"], data["neg_q_pred_lpm"], color=BLUE, linewidth=2, label="NEG q predicted")
    axes[4].set_ylabel("Valve flow [L/min]")
    axes[4].legend(ncol=4, fontsize=8)

    ref_pos_slew = np.r_[0.0, slew_rate(data["ref_pos"], data["time"])]
    ref_neg_slew = np.r_[0.0, slew_rate(data["ref_neg"], data["time"])]
    pressure_pos_slew = np.r_[0.0, slew_rate(data["sen_pos"], data["time"])]
    pressure_neg_slew = np.r_[0.0, slew_rate(data["sen_neg"], data["time"])]
    axes[5].plot(data["time"], ref_pos_slew, color=RED, linestyle="--", linewidth=2, label="REF POS slew")
    axes[5].plot(data["time"], pressure_pos_slew, color=RED, linewidth=2, label="POS slew")
    axes[5].plot(data["time"], ref_neg_slew, color=BLUE, linestyle="--", linewidth=2, label="REF NEG slew")
    axes[5].plot(data["time"], pressure_neg_slew, color=BLUE, linewidth=2, label="NEG slew")
    axes[5].set_ylabel("|dP/dt| [kPa/s]")
    axes[5].set_xlabel("Time [s]")
    axes[5].legend(ncol=4, fontsize=8)

    for axis in axes:
        axis.set_xlabel(axis.get_xlabel(), fontproperties=TIMES)
        axis.set_ylabel(axis.get_ylabel(), fontproperties=TIMES)
        axis.grid(which="major", color="silver", linewidth=1)
        axis.grid(which="minor", color="lightgray", linewidth=0.5)
        axis.minorticks_on()
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontproperties(TIMES)
        for legend_text in axis.get_legend().get_texts():
            legend_text.set_fontproperties(TIMES)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=1.2)
    fig.savefig(output_dir / "trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    console_metrics = {
        "variant": metrics["variant"],
        "steps": metrics["steps"],
        "reward_per_step": round(metrics["reward_per_step"], 5),
        "pos_rmse": round(metrics["pos_rmse"], 4),
        "neg_rmse": round(metrics["neg_rmse"], 4),
        "pos_bias": round(metrics["pos_bias"], 4),
        "neg_bias": round(metrics["neg_bias"], 4),
        "reference_content_match": metrics["reference_content_match"],
        "output_dir": str(output_dir),
    }
    print(f"[ INFO] {model_name}: {console_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=EVAL_STEPS,
        help="evaluation steps; default is 2048 (40.96 s at 50 Hz) for every model",
    )
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument(
        "--tag",
        default=None,
        help="optional output tag, for example legacy_easy, to avoid overwriting another evaluation",
    )
    parser.add_argument(
        "--reference-profile",
        type=Path,
        default=None,
        help="override evaluation reference with a YAML profile without changing the saved model config",
    )
    parser.add_argument(
        "--bank-member",
        type=int,
        default=None,
        help=(
            "evaluate exactly one member of an optimized reference bank; the full bank YAML "
            "still supplies the reference hash used by --require-reference-match"
        ),
    )
    parser.add_argument(
        "--require-reference-match",
        action="store_true",
        help=(
            "fail unless --reference-profile has exactly the same YAML content "
            "as the model's training reference; use for ablation reports"
        ),
    )
    parser.add_argument(
        "--init-pos",
        type=float,
        default=None,
        help="override the model's initial positive pressure for evaluation",
    )
    parser.add_argument(
        "--init-neg",
        type=float,
        default=None,
        help="override the model's initial negative pressure for evaluation",
    )
    args = parser.parse_args()

    cfg_path = MODELS_DIR / args.model / "cfg.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    reference_spec = None
    reference_identity = None
    bank_member_index = None
    bank_member_id = None
    if args.reference_profile is not None:
        with args.reference_profile.open(encoding="utf-8") as handle:
            reference_spec = yaml.safe_load(handle)
        reference_identity = reference_spec
        profile_scale = (reference_spec.get("design") or {}).get("action_scale")
        model_scale = cfg.get("obs", {})
        if profile_scale is not None and len(profile_scale) >= 2:
            if not np.isclose(float(profile_scale[0]), float(model_scale.get("scale_low", 0.8))):
                parser.error(
                    "reference profile action scale does not match model scale: "
                    f"profile={profile_scale}, model=[{model_scale.get('scale_low', 0.8)}, "
                    f"{model_scale.get('scale_high', 1.0)}]"
                )
        if args.require_reference_match:
            training_reference = reference_spec_from_model_config(cfg)
            if reference_sha256(training_reference) != reference_sha256(reference_spec):
                parser.error(
                    "evaluation reference differs from the model training reference; "
                    "omit --require-reference-match only for an explicit transfer test"
                )
        if args.bank_member is not None:
            try:
                reference_spec, bank_member_id = select_optimized_bank_member(reference_spec, args.bank_member)
                bank_member_index = args.bank_member
            except ValueError as exc:
                parser.error(str(exc))
    elif args.bank_member is not None:
        parser.error("--bank-member requires --reference-profile")
    output_tag = args.tag
    if bank_member_index is not None:
        suffix = f"member{bank_member_index:02d}"
        output_tag = f"{output_tag}_{suffix}" if output_tag else suffix
    evaluate(
        args.model,
        args.steps,
        args.tolerance,
        reference_spec,
        output_tag,
        init_pos=args.init_pos,
        init_neg=args.init_neg,
        reference_identity=reference_identity,
        bank_member_index=bank_member_index,
        bank_member_id=bank_member_id,
    )


if __name__ == "__main__":
    main()
