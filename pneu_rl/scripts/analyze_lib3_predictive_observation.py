#!/usr/bin/env python3
"""Diagnose whether lib3 observation groups preserve pressure dynamics.

This is an offline diagnostic, not an RL policy and not a replacement for the
lib3 simulator.  It fits small supervised transition models on one common SAC
replay buffer while changing only the predictor input.  For horizon ``h`` the
models receive the same recorded action sequence ``a[t:t+h]`` and predict the
four-channel pressure change ``P[t+h] - P[t]``.

The default source is the 260810 history+rate+reserve seed-0 run because its
replay contains the base history/action-history observation and the two
derived feature groups.  Rate and reserve are also recomputed from raw pressure
history, and exact reconstruction is audited before training.

The resulting comparison can diagnose:

* current vs pressure history: whether pressure trend is predictive;
* pressure history vs pressure+action history: whether delayed valve commands
  are predictive;
* compact rate/reserve vs full history: whether the physics coordinates act as
  a compact predictive summary;
* history+action vs history+action+rate/reserve: finite-model inductive bias,
  not additional information (the features are deterministic here);
* zero-6 vs rate/reserve: an input-dimension/parameter-count placebo control.

It cannot by itself prove causal necessity or real-system observability.  The
repository documentation describes the required pulse intervention and common
policy-independent dataset for publication-grade evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    ROOT
    / "pneu_rl/models/260810_lib3_ablation_profile_hist_rate_reserve_seed0"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "pneu_rl/exp/lib3_predictive_observation/"
    "260824_hist_rate_reserve_replay_pilot"
)

PRESSURE_NAMES = ("CH_POS", "CH_NEG", "ACT_POS", "ACT_NEG")
FEATURE_SET_ORDER = (
    "current",
    "pressure_history",
    "pressure_action_history",
    "compact_rate_reserve_action_history",
    "history_action_zero2",
    "history_action_reserve",
    "history_action_zero4",
    "history_action_rate",
    "history_action_zero6",
    "history_action_rate_reserve",
)


@dataclass(frozen=True)
class ReplayLayout:
    num_prev: int
    pressure_channels: int
    pressure_history_dim: int
    action_history_len: int
    action_dim: int
    action_history_dim: int
    source_physics_mode: str
    source_physics_dim: int
    observation_without_goal_dim: int
    full_state_dim: int
    training_total_steps: int
    replay_global_start_step: int
    episode_horizon: int
    boundary_source: str


@dataclass
class DatasetArrays:
    pressure_history: np.ndarray
    action_history: np.ndarray
    rate: np.ndarray
    reserve: np.ndarray
    future_actions: np.ndarray
    target_delta_pressure: np.ndarray
    split: np.ndarray
    transition_indices: np.ndarray


class PressureDeltaPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(PRESSURE_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SET_ORDER,
        default=list(FEATURE_SET_ORDER),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=120_000,
        help="Maximum selected transitions per horizon across all splits.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit replay layout and deterministic feature reconstruction only.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def physics_dim(mode: str, enabled: bool) -> int:
    if not enabled:
        return 0
    dims = {
        "none": 0,
        "rate": 4,
        "zero4": 4,
        "reserve": 2,
        "rate_reserve": 6,
        "all": 18,
    }
    if mode not in dims:
        raise ValueError(f"unsupported physics feature mode: {mode!r}")
    return dims[mode]


def load_replay(model_dir: Path) -> tuple[dict, Sequence[tuple], ReplayLayout]:
    cfg_path = model_dir / "cfg.yaml"
    replay_path = model_dir / "buffer.pkl"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # This is a repository-local artifact produced by ReplayBuffer.save_buffer.
    # Never use this loader for untrusted pickle files.
    with replay_path.open("rb") as handle:
        replay = pickle.load(handle)
    if not replay:
        raise ValueError(f"empty replay buffer: {replay_path}")

    env_cfg = cfg["env"]
    num_prev = int(env_cfg["num_prev"])
    action_history_len = int(env_cfg.get("action_history_len", 0))
    source_mode = str(env_cfg.get("physics_feature_mode", "none"))
    source_physics_dim = physics_dim(
        source_mode,
        bool(env_cfg.get("include_physics_features", False)),
    )
    pressure_history_dim = (num_prev + 1) * 4
    action_history_dim = action_history_len * 6
    sample_obs = np.asarray(replay[0][0], dtype=np.float64)
    sample_goal = np.asarray(replay[0][1], dtype=np.float64)
    expected_obs_dim = pressure_history_dim + action_history_dim + source_physics_dim
    if sample_obs.shape != (expected_obs_dim,):
        raise ValueError(
            "This diagnostic currently expects pressure history + action history + "
            "optional rate/reserve only. "
            f"Observed {sample_obs.shape}, expected {(expected_obs_dim,)} from cfg."
        )
    sample_next = np.asarray(replay[0][4], dtype=np.float64)
    expected_full_dim = expected_obs_dim + sample_goal.size
    if sample_next.shape != (expected_full_dim,):
        raise ValueError(
            f"next_state shape {sample_next.shape} != expected {(expected_full_dim,)}"
        )
    masks = np.fromiter((float(item[5]) for item in replay), dtype=np.float64)
    terminal_count = int(np.sum(masks == 0.0))
    episode_horizon = int(cfg.get("model", {}).get("horizon", 0))
    training_total_steps = len(replay)
    boundary_source = "stored terminal mask"
    if terminal_count == 0:
        infos_path = model_dir / "infos.pkl"
        if episode_horizon <= 0 or not infos_path.is_file():
            raise ValueError(
                "replay has no terminal masks; cfg model.horizon and infos.pkl "
                "are required to reconstruct episode groups"
            )
        with infos_path.open("rb") as handle:
            infos = pickle.load(handle)
        if not infos:
            raise ValueError(f"empty training metadata: {infos_path}")
        final_key = max(infos)
        training_total_steps = int(infos[final_key]["steps"])
        boundary_source = "cfg model.horizon + infos.pkl total steps"
    replay_global_start_step = training_total_steps - len(replay)
    if replay_global_start_step < 0:
        raise ValueError(
            f"training_total_steps={training_total_steps} < replay length={len(replay)}"
        )

    layout = ReplayLayout(
        num_prev=num_prev,
        pressure_channels=4,
        pressure_history_dim=pressure_history_dim,
        action_history_len=action_history_len,
        action_dim=6,
        action_history_dim=action_history_dim,
        source_physics_mode=source_mode,
        source_physics_dim=source_physics_dim,
        observation_without_goal_dim=expected_obs_dim,
        full_state_dim=expected_full_dim,
        training_total_steps=training_total_steps,
        replay_global_start_step=replay_global_start_step,
        episode_horizon=episode_horizon,
        boundary_source=boundary_source,
    )
    return cfg, replay, layout


def derive_rate_reserve(
    pressure_history_flat: np.ndarray,
    layout: ReplayLayout,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(pressure_history_flat, dtype=np.float64).reshape(
        layout.num_prev + 1,
        layout.pressure_channels,
    )
    if len(history) < 2:
        rate = np.zeros(4, dtype=np.float64)
    else:
        # lib3 runs at 50 Hz.  cfg validation below records this assumption.
        rate = (history[-1] - history[-2]) * 50.0
    pressure = history[-1]
    reserve = np.asarray(
        [pressure[0] - pressure[2], pressure[3] - pressure[1]],
        dtype=np.float64,
    )
    return rate, reserve


def audit_reconstruction(
    cfg: dict,
    replay: Sequence[tuple],
    layout: ReplayLayout,
    count: int = 2000,
) -> dict:
    freq = float(cfg.get("obs", {}).get("freq", 50.0))
    if not math.isclose(freq, 50.0):
        raise ValueError(
            "derive_rate_reserve currently matches the 50 Hz lib3 definition; "
            f"cfg reports {freq} Hz"
        )
    indices = np.linspace(0, len(replay) - 1, min(count, len(replay)), dtype=int)
    max_rate_error = 0.0
    max_reserve_error = 0.0
    feature_start = layout.pressure_history_dim + layout.action_history_dim
    for index in indices:
        obs = np.asarray(replay[int(index)][0], dtype=np.float64)
        rate, reserve = derive_rate_reserve(obs[: layout.pressure_history_dim], layout)
        if layout.source_physics_mode in ("rate", "rate_reserve", "all"):
            stored_rate = obs[feature_start : feature_start + 4]
            max_rate_error = max(max_rate_error, float(np.max(np.abs(rate - stored_rate))))
        if layout.source_physics_mode == "reserve":
            stored_reserve = obs[feature_start : feature_start + 2]
            max_reserve_error = max(
                max_reserve_error,
                float(np.max(np.abs(reserve - stored_reserve))),
            )
        elif layout.source_physics_mode in ("rate_reserve", "all"):
            stored_reserve = obs[feature_start + 4 : feature_start + 6]
            max_reserve_error = max(
                max_reserve_error,
                float(np.max(np.abs(reserve - stored_reserve))),
            )
    return {
        "checked_samples": int(len(indices)),
        "max_rate_reconstruction_error": max_rate_error,
        "max_reserve_reconstruction_error": max_reserve_error,
        "rate_is_deterministic_from_pressure_history": max_rate_error <= 1e-12,
        "reserve_is_deterministic_from_current_pressure": max_reserve_error <= 1e-12,
    }


def transition_metadata(
    replay: Sequence[tuple],
    layout: ReplayLayout,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.fromiter((float(item[5]) for item in replay), dtype=np.float64)
    if np.any(masks == 0.0):
        episode_ids = np.zeros(len(replay), dtype=np.int64)
        if len(replay) > 1:
            episode_ids[1:] = np.cumsum(masks[:-1] == 0.0)
    else:
        global_steps = layout.replay_global_start_step + np.arange(
            len(replay), dtype=np.int64
        )
        episode_ids = global_steps // layout.episode_horizon
    return masks, episode_ids


def valid_indices_for_horizon(
    masks: np.ndarray,
    episode_ids: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if horizon < 1:
        raise ValueError(f"horizon must be >=1, got {horizon}")
    last_start = len(masks) - horizon
    if last_start < 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.arange(last_start + 1, dtype=np.int64)
    same_episode = episode_ids[candidates] == episode_ids[candidates + horizon - 1]
    if horizon == 1:
        return candidates[same_episode]
    terminal = (masks == 0.0).astype(np.int64)
    prefix = np.r_[0, np.cumsum(terminal)]
    # The terminal transition itself has a valid next_state.  Only boundaries
    # before the last transition of the requested horizon are forbidden.
    crossed = prefix[candidates + horizon - 1] - prefix[candidates]
    return candidates[(crossed == 0) & same_episode]


def split_code(episode_ids: np.ndarray) -> np.ndarray:
    # Episode-grouped 70/10/20 split.  Rows from an episode never cross splits.
    bucket = np.mod(episode_ids, 10)
    return np.where(bucket <= 6, 0, np.where(bucket == 7, 1, 2)).astype(np.int8)


def subsample_by_split(
    valid_indices: np.ndarray,
    episode_ids: np.ndarray,
    max_samples: int,
    seed: int = 73021,
) -> tuple[np.ndarray, np.ndarray]:
    codes = split_code(episode_ids[valid_indices])
    if max_samples <= 0 or len(valid_indices) <= max_samples:
        return valid_indices, codes
    rng = np.random.default_rng(seed)
    ratios = (0.7, 0.1, 0.2)
    selected: list[np.ndarray] = []
    for code, ratio in enumerate(ratios):
        candidates = valid_indices[codes == code]
        take = min(len(candidates), max(1, int(round(max_samples * ratio))))
        chosen = rng.choice(candidates, size=take, replace=False)
        selected.append(np.sort(chosen))
    indices = np.sort(np.concatenate(selected))
    return indices, split_code(episode_ids[indices])


def build_dataset(
    replay: Sequence[tuple],
    layout: ReplayLayout,
    horizon: int,
    max_samples: int,
) -> DatasetArrays:
    masks, episode_ids = transition_metadata(replay, layout)
    valid = valid_indices_for_horizon(masks, episode_ids, horizon)
    selected, splits = subsample_by_split(valid, episode_ids, max_samples)
    if len(selected) == 0:
        raise ValueError(f"no valid transitions for horizon={horizon}")

    n = len(selected)
    pressure_history = np.empty((n, layout.pressure_history_dim), dtype=np.float32)
    action_history = np.empty((n, layout.action_history_dim), dtype=np.float32)
    rate = np.empty((n, 4), dtype=np.float32)
    reserve = np.empty((n, 2), dtype=np.float32)
    future_actions = np.empty((n, horizon * layout.action_dim), dtype=np.float32)
    target = np.empty((n, 4), dtype=np.float32)

    ph_end = layout.pressure_history_dim
    ah_end = ph_end + layout.action_history_dim
    history_shape = (layout.num_prev + 1, layout.pressure_channels)
    for row, start in enumerate(selected):
        obs = np.asarray(replay[int(start)][0], dtype=np.float64)
        ph = obs[:ph_end]
        ah = obs[ph_end:ah_end]
        derived_rate, derived_reserve = derive_rate_reserve(ph, layout)
        pressure_history[row] = ph
        action_history[row] = ah
        rate[row] = derived_rate
        reserve[row] = derived_reserve
        future_actions[row] = np.concatenate(
            [
                np.asarray(replay[int(step)][2], dtype=np.float64).reshape(-1)
                for step in range(int(start), int(start) + horizon)
            ]
        )
        current_pressure = ph.reshape(history_shape)[-1]
        target_state = np.asarray(replay[int(start) + horizon - 1][4], dtype=np.float64)
        future_pressure = target_state[:ph_end].reshape(history_shape)[-1]
        target[row] = future_pressure - current_pressure

    counts = {int(code): int(np.sum(splits == code)) for code in (0, 1, 2)}
    if min(counts.values()) == 0:
        raise ValueError(f"empty train/validation/test split: {counts}")
    return DatasetArrays(
        pressure_history=pressure_history,
        action_history=action_history,
        rate=rate,
        reserve=reserve,
        future_actions=future_actions,
        target_delta_pressure=target,
        split=splits,
        transition_indices=selected,
    )


def feature_matrix(name: str, data: DatasetArrays) -> np.ndarray:
    current = data.pressure_history[:, -4:]
    if name == "current":
        state = current
    elif name == "pressure_history":
        state = data.pressure_history
    elif name == "pressure_action_history":
        state = np.c_[data.pressure_history, data.action_history]
    elif name == "compact_rate_reserve_action_history":
        state = np.c_[current, data.rate, data.reserve, data.action_history]
    elif name == "history_action_zero2":
        state = np.c_[
            data.pressure_history,
            data.action_history,
            np.zeros((len(current), 2), dtype=np.float32),
        ]
    elif name == "history_action_reserve":
        state = np.c_[data.pressure_history, data.action_history, data.reserve]
    elif name == "history_action_zero4":
        state = np.c_[
            data.pressure_history,
            data.action_history,
            np.zeros((len(current), 4), dtype=np.float32),
        ]
    elif name == "history_action_rate":
        state = np.c_[data.pressure_history, data.action_history, data.rate]
    elif name == "history_action_zero6":
        state = np.c_[
            data.pressure_history,
            data.action_history,
            np.zeros((len(current), 6), dtype=np.float32),
        ]
    elif name == "history_action_rate_reserve":
        state = np.c_[
            data.pressure_history,
            data.action_history,
            data.rate,
            data.reserve,
        ]
    else:
        raise ValueError(f"unknown feature set: {name}")
    # Known recorded action sequence is a controlled input shared by every
    # feature set.  It is not counted as observation evidence.
    return np.c_[state, data.future_actions].astype(np.float32, copy=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    seed_everything(seed)
    train_mask = split == 0
    val_mask = split == 1
    test_mask = split == 2
    x_mean, x_std = standardizer(x[train_mask])
    y_mean, y_std = standardizer(y[train_mask])
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    y_norm = ((y - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_mask]),
        torch.from_numpy(y_norm[train_mask]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    model = PressureDeltaPredictor(x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0

    val_x = torch.from_numpy(x_norm[val_mask]).to(device)
    val_y = torch.from_numpy(y_norm[val_mask]).to(device)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("predictor training did not produce a checkpoint")
    model.load_state_dict(best_state)

    pred_norm = predict_batches(model, x_norm[test_mask], batch_size, device)
    pred = pred_norm * y_std + y_mean
    truth = y[test_mask]
    error = pred - truth
    channel_rmse = np.sqrt(np.mean(np.square(error), axis=0))
    aggregate_rmse = float(np.sqrt(np.mean(np.square(error))))
    persistence_rmse = float(np.sqrt(np.mean(np.square(truth))))
    metrics = {
        "seed": int(seed),
        "input_dim": int(x.shape[1]),
        "train_samples": int(np.sum(train_mask)),
        "validation_samples": int(np.sum(val_mask)),
        "test_samples": int(np.sum(test_mask)),
        "best_epoch": int(best_epoch + 1),
        "best_validation_normalized_mse": float(best_val),
        "aggregate_rmse_kpa": aggregate_rmse,
        "persistence_rmse_kpa": persistence_rmse,
        "skill_vs_persistence": float(1.0 - aggregate_rmse / max(persistence_rmse, 1e-12)),
    }
    for index, channel in enumerate(PRESSURE_NAMES):
        metrics[f"{channel.lower()}_rmse_kpa"] = float(channel_rmse[index])
    return metrics, np.c_[truth, pred]


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(choice)


def write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((int(row["horizon_steps"]), str(row["feature_set"])), []).append(row)
    result: list[dict] = []
    for (horizon, feature), group in sorted(groups.items()):
        rmse = np.asarray([item["aggregate_rmse_kpa"] for item in group], dtype=np.float64)
        result.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": horizon / 50.0,
                "feature_set": feature,
                "predictor_seed_count": len(group),
                "aggregate_rmse_mean_kpa": float(np.mean(rmse)),
                "aggregate_rmse_std_kpa": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
            }
        )
    return result


def plot_aggregate(rows: list[dict], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    features = list(dict.fromkeys(row["feature_set"] for row in rows))
    for feature in features:
        selected = [row for row in rows if row["feature_set"] == feature]
        selected.sort(key=lambda row: row["horizon_steps"])
        ax.errorbar(
            [row["horizon_seconds"] for row in selected],
            [row["aggregate_rmse_mean_kpa"] for row in selected],
            yerr=[row["aggregate_rmse_std_kpa"] for row in selected],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=feature,
        )
    ax.set_xlabel("Prediction horizon [sec]")
    ax.set_ylabel("Four-pressure prediction RMSE [kPa]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if any(horizon < 1 for horizon in args.horizons):
        raise ValueError("all horizons must be positive")
    torch.set_num_threads(max(int(args.torch_threads), 1))
    cfg, replay, layout = load_replay(args.model_dir.resolve())
    audit = audit_reconstruction(cfg, replay, layout)
    report = {
        "purpose": "offline observation predictiveness diagnostic; not an RL policy or simulator",
        "source_model_dir": str(args.model_dir.resolve()),
        "replay_transition_count": len(replay),
        "layout": asdict(layout),
        "audit": audit,
        "split": "episode-grouped 70/10/20 by episode_id modulo 10",
        "boundary_source": layout.boundary_source,
        "target": "four-pressure delta P[t+h]-P[t] in kPa",
        "future_control": "recorded action sequence a[t:t+h], shared by every feature set",
        "limitations": [
            "source replay is on-policy and therefore preliminary evidence",
            "rate/reserve are deterministic from the base history/current pressures",
            "prediction gain of explicit rate/reserve indicates finite-model inductive bias, not new information",
            "causal necessity requires matched simulator/real pulse interventions",
        ],
    }
    print(json.dumps(report, indent=2))
    if args.check_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rows: list[dict] = []
    for horizon in args.horizons:
        data = build_dataset(replay, layout, horizon, args.max_samples)
        print(
            f"[h={horizon:>2}] samples={len(data.split)} "
            f"train/val/test={[(data.split == code).sum() for code in (0, 1, 2)]}"
        )
        for feature in args.feature_sets:
            x = feature_matrix(feature, data)
            for seed in args.seeds:
                metrics, _ = train_one(
                    x=x,
                    y=data.target_delta_pressure,
                    split=data.split,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                metrics = {
                    "horizon_steps": int(horizon),
                    "horizon_seconds": float(horizon / 50.0),
                    "feature_set": feature,
                    **metrics,
                }
                rows.append(metrics)
                print(
                    f"  {feature:<39} seed={seed} "
                    f"RMSE={metrics['aggregate_rmse_kpa']:.5f} kPa"
                )
                write_metrics(args.output_dir / "metrics.csv", rows)

    aggregate = aggregate_rows(rows)
    write_metrics(args.output_dir / "aggregate_metrics.csv", aggregate)
    report.update(
        {
            "device": str(device),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "aggregate_metrics": aggregate,
        }
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if not args.no_plot:
        plot_aggregate(aggregate, args.output_dir / "prediction_rmse.png")
    print(f"saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
