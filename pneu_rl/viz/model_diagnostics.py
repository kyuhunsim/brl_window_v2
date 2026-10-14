"""Create reproducible training-diagnostic plots for every saved RL model.

The script intentionally uses ``train_diagnostics.csv`` rather than loading
``model.pth``/``infos.pkl``.  The CSV is sufficient for reward, tracking,
chamber, and valve-behaviour analysis and keeps plotting independent of the
training runtime.

Run from the repository root with the project environment, for example:

    python pneu_rl/viz/model_diagnostics.py

Outputs are written to ``img/model_diagnostics`` by default.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "pneu_rl" / "models"


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_infos(path: Path) -> list[dict[str, float]]:
    """Read the legacy logger format when train diagnostics are unavailable."""
    with path.open("rb") as handle:
        infos = pickle.load(handle)
    rows = []
    for episode, info in infos.items():
        row = {"epi": float(episode), "reward": float(info.get("reward", float("nan")))}
        for key in ("temporal_weight", "alpha"):
            value = info.get(key)
            if value is not None:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
        rows.append(row)
    return sorted(rows, key=lambda row: row["epi"])


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values.copy()
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def safe_filename(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", name).strip("_")
    return value or "model"


def model_features(cfg: dict, retrain_cfg: dict | None) -> dict[str, object]:
    env = cfg.get("env", {}) or {}
    model = cfg.get("model", {}) or {}
    reward = env.get("rwd_kwargs", {}) or {}
    pred = cfg.get("pred")
    retrain_reward = ((retrain_cfg or {}).get("env", {}) or {}).get("rwd_kwargs", {}) or {}

    temporal = float(model.get("temporal_weight", 0) or 0)
    spatial = float(model.get("spatial_weight", 0) or 0)
    return {
        "pid": cfg.get("pid") is not None,
        "caps": temporal != 0 or spatial != 0,
        "mpobs": pred is not None and int(env.get("num_prev", 0) or 0) > 0 and int(env.get("num_pred", 0) or 0) > 1,
        "history": int(env.get("num_prev", 0) or 0) > 0,
        "prediction": pred is not None,
        "prev_action": bool(env.get("include_prev_action_obs", False)),
        "diff_velocity": bool(env.get("include_pressure_diff_velocity_obs", False)),
        "action_delta": float(retrain_reward.get("action_delta_rwd_coeff", reward.get("action_delta_rwd_coeff", 0)) or 0),
        "conflict_penalty": float(retrain_reward.get("conflict_rwd_coeff", reward.get("conflict_rwd_coeff", 0)) or 0),
        "temporal_weight": temporal,
        "spatial_weight": spatial,
        "retrain_epi": int(cfg.get("epi", 0) or 0) if retrain_cfg else None,
        "parent_model": (retrain_cfg or {}).get("parent_model"),
        "retrain_type": (retrain_cfg or {}).get("type"),
    }


def phase_summary(rows: list[dict[str, float]], start: int, end: int) -> dict[str, float]:
    phase = rows[start:end]

    def mean(key: str) -> float:
        values = [float(row[key]) for row in phase if key in row and isinstance(row[key], (int, float))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "episodes": float(len(phase)),
        "epi_start": float(phase[0]["epi"]) if phase else float("nan"),
        "epi_end": float(phase[-1]["epi"]) if phase else float("nan"),
        "reward": mean("reward"),
        "pos_rmse": mean("pos_rmse"),
        "neg_rmse": mean("neg_rmse"),
        "pos_bias": mean("pos_bias"),
        "neg_bias": mean("neg_bias"),
        "pos_conflict": mean("pos_conflict_mean"),
        "neg_conflict": mean("neg_conflict_mean"),
        "ch_pos_end": mean("ch_pos_end"),
        "ch_neg_end": mean("ch_neg_end"),
        "ch_pos_max": mean("ch_pos_max"),
        "ch_neg_min": mean("ch_neg_min"),
        "pos_action_delta": mean("u_pos_in_delta") + mean("u_pos_out_delta"),
        "neg_action_delta": mean("u_neg_in_delta") + mean("u_neg_out_delta"),
    }


def classify(pre: dict[str, float], post: dict[str, float] | None) -> str:
    current = post or pre
    if not np.isfinite(current["pos_rmse"]) or not np.isfinite(current["neg_rmse"]):
        return "reward-only"
    good = current["pos_rmse"] < 5 and current["neg_rmse"] < 5
    partial = current["pos_rmse"] < 7 and current["neg_rmse"] < 7
    if good:
        return "success"
    if partial:
        return "partial"
    return "failure"


def load_models() -> list[dict]:
    models = []
    for model_dir in sorted(MODELS_DIR.iterdir(), key=lambda path: path.name):
        csv_path = model_dir / "train_diagnostics.csv"
        infos_path = model_dir / "infos.pkl"
        cfg_path = model_dir / "cfg.yaml"
        if not cfg_path.exists() or (not csv_path.exists() and not infos_path.exists()):
            continue
        cfg = read_yaml(cfg_path)
        retrain_path = model_dir / "retrain_cfg.yaml"
        retrain_cfg = read_yaml(retrain_path) if retrain_path.exists() else None
        diagnostics = csv_path.exists()
        rows = read_csv(csv_path) if diagnostics else read_infos(infos_path)
        retrain_epi = int(cfg.get("epi", 0) or 0) if retrain_cfg and len(rows) > int(cfg.get("epi", 0) or 0) else None
        split = next((index for index, row in enumerate(rows) if retrain_epi is not None and row["epi"] > retrain_epi), len(rows))
        pre = phase_summary(rows, max(0, split - 100) if retrain_epi is not None else max(0, len(rows) - 100), split if retrain_epi is not None else len(rows))
        post = phase_summary(rows, max(split, len(rows) - 100), len(rows)) if retrain_epi is not None else None
        models.append({
            "name": model_dir.name,
            "dir": model_dir,
            "rows": rows,
            "cfg": cfg,
            "retrain_cfg": retrain_cfg,
            "features": model_features(cfg, retrain_cfg),
            "retrain_epi": retrain_epi,
            "split": split,
            "pre": pre,
            "post": post,
            "classification": classify(pre, post),
            "diagnostics": diagnostics,
        })
    return models


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_retrain_marker(ax: plt.Axes, model: dict) -> None:
    retrain_epi = model["retrain_epi"]
    if retrain_epi is None:
        return
    ax.axvline(retrain_epi, color="red", linestyle="--", linewidth=1.6, label=f"retrain @ {retrain_epi:g}")
    ax.axvspan(retrain_epi, max(row["epi"] for row in model["rows"]), color="red", alpha=0.045)


def plot_model(model: dict, output_dir: Path) -> Path:
    rows = model["rows"]
    episode = np.asarray([row["epi"] for row in rows], dtype=float)
    smooth_window = max(7, min(51, len(rows) // 20))
    reward = np.asarray([row["reward"] for row in rows], dtype=float)
    def series(key: str) -> np.ndarray:
        return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)

    pos_rmse = series("pos_rmse")
    neg_rmse = series("neg_rmse")
    pos_bias = series("pos_bias")
    neg_bias = series("neg_bias")
    pos_conflict = series("pos_conflict_mean")
    neg_conflict = series("neg_conflict_mean")
    ch_pos = series("ch_pos_end")
    ch_neg = series("ch_neg_end")
    pos_delta = series("u_pos_in_delta") + series("u_pos_out_delta")
    neg_delta = series("u_neg_in_delta") + series("u_neg_out_delta")

    # Keep generated PNGs portable even on machines without a Korean font.
    title = model["name"].encode("ascii", "replace").decode("ascii")
    features = model["features"]
    tags = [
        f"PID={'on' if features['pid'] else 'off'}",
        f"CAPS={'on' if features['caps'] else 'off'}",
        f"MPOBS={'on' if features['mpobs'] else 'off'}",
    ]

    if not model["diagnostics"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episode, reward, color="#b8c0cc", linewidth=0.7, label="episode reward")
        ax.plot(episode, rolling(reward, smooth_window), color="#123b67", linewidth=2, label=f"rolling mean ({smooth_window})")
        ax.set_title(f"{title} | " + ", ".join(tags), fontsize=16, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(loc="best", fontsize=8)
        style_axes(ax)
        add_retrain_marker(ax, model)
        ax.set_xlim(float(episode.min()), float(episode.max()))
        ax.text(0.01, 0.02, "reward-only | no train_diagnostics.csv", transform=ax.transAxes, fontsize=9, color="#444444")
        fig.tight_layout()
        output = output_dir / f"model_{safe_filename(model['name'])}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    axes = axes.ravel()
    fig.suptitle(f"{title} | " + ", ".join(tags), fontsize=16, fontweight="bold")

    axes[0].plot(episode, reward, color="#b8c0cc", linewidth=0.7, label="episode reward")
    axes[0].plot(episode, rolling(reward, smooth_window), color="#123b67", linewidth=2, label=f"rolling mean ({smooth_window})")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(episode, pos_rmse, color="#d1495b", alpha=0.28, linewidth=0.7)
    axes[1].plot(episode, neg_rmse, color="#147d92", alpha=0.28, linewidth=0.7)
    axes[1].plot(episode, rolling(pos_rmse, smooth_window), color="#d1495b", linewidth=2, label="pos RMSE")
    axes[1].plot(episode, rolling(neg_rmse, smooth_window), color="#147d92", linewidth=2, label="neg RMSE")
    axes[1].set_ylabel("Tracking RMSE [kPa]")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(episode, rolling(pos_bias, smooth_window), color="#d1495b", linewidth=2, label="pos bias")
    axes[2].plot(episode, rolling(neg_bias, smooth_window), color="#147d92", linewidth=2, label="neg bias")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Tracking bias [kPa]")
    axes[2].legend(loc="best", fontsize=8)

    axes[3].plot(episode, rolling(pos_conflict, smooth_window), color="#d1495b", linewidth=2, label="pos conflict")
    axes[3].plot(episode, rolling(neg_conflict, smooth_window), color="#147d92", linewidth=2, label="neg conflict")
    axes[3].set_ylabel("Valve conflict mean")
    axes[3].set_ylim(bottom=0)
    axes[3].legend(loc="best", fontsize=8)

    axes[4].plot(episode, rolling(ch_pos, smooth_window), color="#d1495b", linewidth=2, label="chamber pos end")
    axes[4].plot(episode, rolling(ch_neg, smooth_window), color="#147d92", linewidth=2, label="chamber neg end")
    axes[4].set_ylabel("Chamber end pressure [kPa]")
    axes[4].legend(loc="best", fontsize=8)

    axes[5].plot(episode, rolling(pos_delta, smooth_window), color="#d1495b", linewidth=2, label="pos action delta")
    axes[5].plot(episode, rolling(neg_delta, smooth_window), color="#147d92", linewidth=2, label="neg action delta")
    axes[5].set_ylabel("Valve action delta")
    axes[5].set_xlabel("Episode")
    axes[5].legend(loc="best", fontsize=8)

    for ax in axes:
        style_axes(ax)
        add_retrain_marker(ax, model)
    axes[0].set_xlim(float(episode.min()), float(episode.max()))
    phase_label = "post-retrain" if model["post"] else "last100"
    axes[0].text(
        0.01,
        0.02,
        f"{model['classification']} | summary={phase_label}",
        transform=axes[0].transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = output_dir / f"model_{safe_filename(model['name'])}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_comparison(models: list[dict], output_dir: Path) -> Path:
    labels = [model["name"].encode("ascii", "replace").decode("ascii") for model in models]
    summaries = [model["post"] or model["pre"] for model in models]
    x = np.arange(len(models))
    colors = ["#2a9d8f" if model["classification"] == "success" else "#e9c46a" if model["classification"] == "partial" else "#e76f51" for model in models]

    fig, axes = plt.subplots(2, 2, figsize=(19, 12))
    plots = [
        ("reward", "Final-phase reward", "Reward"),
        ("pos_rmse", "Final-phase positive tracking", "RMSE [kPa]"),
        ("neg_rmse", "Final-phase negative tracking", "RMSE [kPa]"),
        ("pos_conflict", "Final-phase valve conflict", "Conflict mean"),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), plots):
        values = [summary[key] for summary in summaries]
        ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=8)
        style_axes(ax)
        for index, model in enumerate(models):
            if model["retrain_epi"] is not None:
                ax.text(index, values[index], "R", color="red", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle("Model comparison (last 100 episodes; retrained models use post-retrain phase)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = output_dir / "model_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_feature_scatter(models: list[dict], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {False: "#457b9d", True: "#d1495b"}
    markers = {False: "o", True: "s"}
    for model in models:
        summary = model["post"] or model["pre"]
        if not np.isfinite(summary["pos_rmse"]) or not np.isfinite(summary["neg_rmse"]):
            continue
        feature = model["features"]
        ax.scatter(
            summary["pos_rmse"],
            summary["neg_rmse"],
            s=100,
            color=colors[bool(feature["pid"])],
            marker=markers[bool(feature["caps"])],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        display_name = model["name"].encode("ascii", "replace").decode("ascii")
        ax.annotate(display_name, (summary["pos_rmse"], summary["neg_rmse"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axvline(5, color="#2a9d8f", linestyle=":", linewidth=1)
    ax.axhline(5, color="#2a9d8f", linestyle=":", linewidth=1)
    ax.text(0.02, 0.98, "Inside dotted lines: pos/neg RMSE < 5 kPa", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("Positive actuator RMSE [kPa]")
    ax.set_ylabel("Negative actuator RMSE [kPa]")
    ax.set_title("Feature comparison: color=PID (blue off / red on), marker=CAPS (circle off / square on)")
    style_axes(ax)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output = output_dir / "feature_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def write_summary(models: list[dict], output_dir: Path) -> Path:
    fields = [
        "model", "classification", "pid", "caps", "mpobs", "history", "prediction", "prev_action", "diff_velocity",
        "action_delta_coeff", "conflict_penalty", "retrain_epi", "phase", "reward", "pos_rmse", "neg_rmse", "pos_bias", "neg_bias",
        "pos_conflict", "neg_conflict", "ch_pos_end", "ch_neg_end",
    ]
    output = output_dir / "model_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            summary = model["post"] or model["pre"]
            feature = model["features"]
            writer.writerow({
                "model": model["name"],
                "classification": model["classification"],
                "pid": int(feature["pid"]),
                "caps": int(feature["caps"]),
                "mpobs": int(feature["mpobs"]),
                "history": int(feature["history"]),
                "prediction": int(feature["prediction"]),
                "prev_action": int(feature["prev_action"]),
                "diff_velocity": int(feature["diff_velocity"]),
                "action_delta_coeff": feature["action_delta"],
                "conflict_penalty": feature["conflict_penalty"],
                "retrain_epi": model["retrain_epi"] or "",
                "phase": "post-retrain" if model["post"] else "last100",
                **{key: f"{summary[key]:.8f}" for key in fields if key in summary},
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "img" / "model_diagnostics")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models()
    if not models:
        raise SystemExit(f"No model diagnostics found under {MODELS_DIR}")
    for model in models:
        plot_model(model, args.output_dir)
    plot_comparison(models, args.output_dir)
    plot_feature_scatter(models, args.output_dir)
    write_summary(models, args.output_dir)
    print(f"Generated {len(models)} model plots in {args.output_dir}")


if __name__ == "__main__":
    main()
