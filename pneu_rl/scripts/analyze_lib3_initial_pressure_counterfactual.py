"""SIM ONLY: replay two frozen lib3 policies from five pressure initializations.

The four non-atmospheric initializations are read from the first real CSV row
whose reference is valid, not a verified pre-first-action measurement. These
rollouts test pressure initialization under a common simulator reset rule;
changing source pressure also changes the reset pump pressures. They do not
reproduce hardware hidden state, manual-start clocks, timing, or physical
mismatch. Every run uses a separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
REAL_ROOT = ROOT / "pneu_rl/exp/drive_zeroshot_20261015"
DEFAULT_OUTPUT = ROOT / "pneu_rl/exp/lib3_initial_pressure_counterfactual_261015"
MODELS = {
    "H_smooth": "260831_lib3_histreward2x2_r025_H_smooth_seed0",
    "HR_smooth": "260831_lib3_histreward2x2_r025_HR_smooth_seed0",
}
SELECTED_2000_SHA256 = {
    "H_smooth": "157cf44004d02b70816f9ad36499855d37a60ab6544dae916284625766fb0879",
    "HR_smooth": "1d966ad5d3e1efaa7c203ad207b9e0d7eab54d3b6f8f95d90d94584796b93dbc",
}
PRESSURE_COLUMNS = ["press_pos", "press_neg", "act_pos_press", "act_neg_press"]
INIT_KEYS = [
    "init_pos_press", "init_neg_press", "init_act_pos_press", "init_act_neg_press"
]
ATM = 101.325
HIDDEN_RESET = (
    "Simulator set_init_env_act resets six valve states; pump phase is zero; "
    "two internal pump pressures use the simulator's initialization formulas "
    "from source pressures, so their values change when source pressure changes. "
    "The reset RULE is common; hidden state values are not held constant. "
    "These hidden states are not measured real states."
)
SOURCE_MEASUREMENT_CAVEAT = (
    "First logged measurement whose reference is valid; not verified as the "
    "physical state immediately before the first policy action."
)
HISTORY_CAVEATS = {
    "fake_atmosphere_history": (
        "Approximate synthetic initial current policy-pressure observation AND "
        "pressure history; not a reproduction of real manual-start timing or "
        "deployment state construction. Plant/sensor initialization retains "
        "the case pressures."
    ),
    "consistent_history": (
        "Initial current policy-pressure observation and pressure history are "
        "filled consistently with the supplied plant pressure; action history "
        "uses the normal environment reset."
    ),
}
SCOPE = (
    "The interventions are pressure initialization and optional synthetic "
    "initial current policy-pressure observation plus pressure history. "
    "Nominal simulator parameters, 50 Hz loop, a common hidden-state RESET "
    "RULE, and fresh member-0 reference are retained. Internal reset pump "
    "pressures vary with source pressure under that rule. Source CSV states "
    "are first valid-reference measurements, not verified pre-first-action "
    "states. These rollouts cannot establish that initialization explains "
    "observed real failures."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_checkpoint(model_dir: Path, model: str) -> Path:
    """Accept the selected episode-2000 file or its compact-ZIP model.pth copy."""
    checkpoint = model_dir / "model_epi_2000.pth"
    if not checkpoint.is_file():
        checkpoint = model_dir / "model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no selected checkpoint in {model_dir}")
    actual = digest(checkpoint)
    if actual != SELECTED_2000_SHA256[model]:
        raise ValueError(
            f"{checkpoint} is not the selected episode-2000 file: SHA256 {actual}"
        )
    return checkpoint


def load_cases() -> dict:
    cases = {
        "atmospheric": {
            "pressure_kpa": [ATM] * 4,
            "source_csv": None,
            "source_row_index": None,
            "source_logged_time_sec": None,
        }
    }
    for csv_path in sorted(REAL_ROOT.glob("*/*.csv")):
        frame = pd.read_csv(csv_path)
        valid = frame[(frame.ref_act_pos > 1.0) & (frame.ref_act_neg > 1.0)]
        if valid.empty:
            raise ValueError(f"no valid reference in {csv_path}")
        row = valid.iloc[0]
        stamp = csv_path.parent.name.split("_260831_", 1)[0]
        cases[stamp] = {
            "pressure_kpa": row[PRESSURE_COLUMNS].to_numpy(float).tolist(),
            "source_csv": str(csv_path.relative_to(ROOT)),
            "source_csv_sha256": digest(csv_path),
            "source_row_index": int(row.name),
            "source_logged_time_sec": float(row.curr_time),
            "source_measurement_caveat": SOURCE_MEASUREMENT_CAVEAT,
        }
    if len(cases) != 5:
        raise ValueError(f"expected atmospheric plus four real cases, got {list(cases)}")
    return cases


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"samples": 0}
    errors = frame[["act_pos", "act_neg"]].to_numpy(float) - frame[
        ["ref_pos", "ref_neg"]
    ].to_numpy(float)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    return {
        "samples": len(frame),
        "rmse_pos_kpa": float(rmse[0]),
        "rmse_neg_kpa": float(rmse[1]),
        "combined_rmse_kpa": float(np.sqrt(np.mean(errors**2))),
        "bias_pos_obs_minus_ref_kpa": float(errors[:, 0].mean()),
        "bias_neg_obs_minus_ref_kpa": float(errors[:, 1].mean()),
        "positive_reserve_mean_kpa": float((frame.chamber_pos - frame.act_pos).mean()),
        "negative_reserve_mean_kpa": float((frame.act_neg - frame.chamber_neg).mean()),
    }


def worker(args: argparse.Namespace) -> None:
    # These are exclusively simulator classes. No real/serial/ROS object exists.
    import torch
    from pneu_env.env3 import PneuEnv3
    from pneu_env.sim3 import PneuSim
    from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
    from pneu_rl.sac_with_loss import SAC
    from evaluate_lib3_screening import reset_to_reference_member, run_episode

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    cases = load_cases()
    case = cases[args.case]
    pressure = np.asarray(case["pressure_kpa"], dtype=float)
    model_name = MODELS[args.model]
    model_dir = ROOT / "pneu_rl/models" / model_name
    cfg_path = model_dir / "cfg.yaml"
    checkpoint = resolve_checkpoint(model_dir, args.model)
    before = {str(p.relative_to(ROOT)): digest(p) for p in (cfg_path, checkpoint)}
    cfg = yaml.full_load(cfg_path.read_text(encoding="utf-8"))
    if cfg["lib"] != 3 or cfg["env"].get("action_chunk_steps", 1) != 1:
        raise ValueError("only frozen lib3 K=1 checkpoints are supported")
    if cfg["obs"]["freq"] != 50 or cfg.get("pred") is not None or cfg.get("pid"):
        raise ValueError("expected 50 Hz, no prediction model, no PID")

    obs_cfg = dict(cfg["obs"])
    obs_cfg.update(dict(zip(INIT_KEYS, pressure.tolist())))
    obs = PneuSim(**obs_cfg)
    reference = make_pressure_reference(reference_spec_from_model_config(cfg))
    env_cfg = dict(cfg["env"])
    env_cfg["verbose"] = False
    env = PneuEnv3(obs=obs, ref=reference, pred=None, **env_cfg)
    model = SAC(env=env, **cfg["model"])
    # No logger, buffer load, training call, optimizer update, or model save.
    model.load_model(path=str(checkpoint), evaluate=True)
    state, _ = reset_to_reference_member(env, 0)
    mode = "consistent_history"
    if args.fake_history:
        mode = "fake_atmosphere_history"
        # Approximate an all-atmospheric initial current policy observation AND
        # pressure history while plant/sensor initialization retain case pressure.
        # The first policy input is atmospheric; the first measurement then
        # replaces one sample. This is NOT a reconstruction of hardware reset.
        env.obs_traj[:] = ATM
        env.curr_full_press[:] = ATM
        env.prev_full_press[:] = ATM
        state, _ = env._state_from_current(env.last_policy_action)
    initial_policy_state = np.asarray(state, dtype=float).tolist()
    initial_valve_debug = obs.get_valve_debug()
    try:
        frame = run_episode(env, model, 2048, state=state)
    finally:
        env.close()
    finite_columns = [
        "time", "ref_pos", "ref_neg", "act_pos", "act_neg", "chamber_pos", "chamber_neg"
    ] + [f"ctrl{index + 1}" for index in range(6)]
    if len(frame) != 2048 or not np.isfinite(frame[finite_columns].to_numpy(float)).all():
        raise RuntimeError("rollout is incomplete or contains nonfinite values")
    frame["control_elapsed_sec"] = (np.arange(len(frame)) + 1) / 50.0
    output_dir = args.output_root / args.model / f"{args.case}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "rollout.csv", index=False)
    after = {str(p.relative_to(ROOT)): digest(p) for p in (cfg_path, checkpoint)}
    if before != after:
        raise RuntimeError("input model/config changed during evaluation")
    result = {
        "model": args.model,
        "model_name": model_name,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "input_sha256": before,
        "original_model_files_unchanged": before == after,
        "case": args.case,
        "initialization": case,
        "initial_pressure_order": PRESSURE_COLUMNS,
        "history_mode": mode,
        "history_caveat": HISTORY_CAVEATS[mode],
        "hidden_reset_assumption": HIDDEN_RESET,
        "initial_valve_debug": {k: float(v) for k, v in initial_valve_debug.items()},
        "initial_policy_state": initial_policy_state,
        "reference_bank": cfg["reference"]["kwargs"]["trajectory_npz"],
        "reference_member": 0,
        "reference_clock_start_sec": 0,
        "frequency_hz": 50,
        "action_chunk_steps": 1,
        "control_steps": 2048,
        "control_duration_sec": 40.96,
        "sensor_logged_time_last_sec": float(frame.time.iloc[-1]),
        "metric_window_clock": "control_elapsed_sec: exact applied simulator steps / 50",
        "metrics": {
            "whole": metrics(frame),
            "first_5_sec": metrics(frame[frame.control_elapsed_sec <= 5]),
            "after_5_sec": metrics(frame[frame.control_elapsed_sec > 5]),
        },
        "scope": SCOPE,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.model, args.case, mode, result["metrics"]["whole"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-fake-history", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=list(MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--fake-history", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output_root = args.output_root.resolve()
    if args.worker:
        worker(args)
        return
    cases = load_cases()
    worker_env = dict(os.environ)
    worker_env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    worker_env.setdefault("MPLCONFIGDIR", "/tmp/brl_lib3_initial_pressure_mpl")
    Path(worker_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    selected = []
    for model in MODELS:
        for case_name, case in cases.items():
            history_modes = [False]
            if args.include_fake_history and case.get("source_row_index") == 1:
                history_modes.append(True)
            for fake in history_modes:
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--model", model, "--case", case_name,
                    "--output-root", str(args.output_root),
                ]
                if fake:
                    command.append("--fake-history")
                subprocess.run(command, check=True, cwd=ROOT, env=worker_env)
                mode = "fake_atmosphere_history" if fake else "consistent_history"
                summary_path = args.output_root / model / f"{case_name}_{mode}" / "summary.json"
                selected.append(json.loads(summary_path.read_text(encoding="utf-8")))
    aggregate = {
        "simulation_only": True,
        "training": False,
        "cases": cases,
        "run_count": len(selected),
        "results": selected,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat = []
    for run in selected:
        row = {k: run[k] for k in ("model", "case", "history_mode")}
        for window, values in run["metrics"].items():
            row.update({f"{window}_{key}": value for key, value in values.items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(args.output_root / "summary.csv", index=False)
    print(f"Completed {len(selected)} simulation-only rollouts: {args.output_root}")


if __name__ == "__main__":
    main()
