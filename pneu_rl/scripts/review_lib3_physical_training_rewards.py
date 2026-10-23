"""Audit recorded SAC learning curves without modifying models or running jobs."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/brl_reward_review_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"H": "#3f6da8", "HR": "#d87927"}
RUNS = [(typ, policy) for typ in (1, 2) for policy in ("H", "HR")]


def scalar(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def read_pickle(path: Path):
    # The logger rewrites this file each episode. Retry a partial live write.
    for attempt in range(8):
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except (EOFError, pickle.UnpicklingError):
            if attempt == 7:
                raise
            time.sleep(0.15)


def config_differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                result.append(path)
            else:
                result.extend(config_differences(a[key], b[key], path))
        return result
    return [] if a == b else [prefix]


def window_stats(frame: pd.DataFrame) -> dict:
    keys = [
        "reward", "pos_rmse", "neg_rmse", "pos_bias", "neg_bias",
        "critic_loss", "policy_loss", "alpha", "action_std_mean",
        "action_saturation_ratio", "act_pos_sign_change_per_sec",
        "act_neg_sign_change_per_sec", "ch_pos_sign_change_per_sec",
        "ch_neg_sign_change_per_sec", "ch_pos_max", "ch_neg_min",
        "u_ch_pos_delta", "u_ch_neg_delta", "u_pos_in_delta",
        "u_pos_out_delta", "u_neg_in_delta", "u_neg_out_delta",
    ]
    result = {"n": len(frame)}
    if not len(frame):
        return result
    result.update(episode_first=int(frame.epi.min()), episode_last=int(frame.epi.max()))
    for key in keys:
        if key not in frame:
            continue
        values = frame[key].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values):
            result[key] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values)),
                "q05": float(np.quantile(values, 0.05)), "q95": float(np.quantile(values, 0.95)),
            }
    if len(frame) > 1:
        result["reward_slope_per_100_episodes"] = float(
            np.polyfit(frame.epi.to_numpy(float), frame.reward.to_numpy(float), 1)[0] * 100
        )
    return result


def read_run(typ: int, policy: str):
    name = f"261023_lib3_opening_type{typ}_{policy}_smooth_seed0"
    folder = ROOT / "pneu_rl/models" / name
    cfg = yaml.safe_load((folder / "cfg.yaml").read_text())
    wall = json.loads((folder / "wall_time.json").read_text())
    infos = read_pickle(folder / "infos.pkl")
    rows = []
    for epi, values in sorted(infos.items()):
        row = {"epi": int(epi)}
        for key in ("reward", "steps", "critic_loss", "policy_loss"):
            if values.get(key) is not None:
                row[key] = scalar(values[key])
        rows.append(row)
    info = pd.DataFrame(rows)
    diag = pd.read_csv(folder / "train_diagnostics.csv")
    exploration = pd.read_csv(folder / "exploration_diagnostics.csv")
    overlap = diag.merge(info[["epi", "reward"]], on="epi", suffixes=("_diag", "_infos"))
    reward_agrees = bool(np.allclose(overlap.reward_diag, overlap.reward_infos, atol=1e-9, rtol=1e-9))
    frame = diag.merge(info.drop(columns="reward"), on="epi", how="inner")
    frame = frame.merge(exploration, on="epi", how="left")
    warmup_episodes = int(cfg["model"].get("start_epi", 10))
    carry_index = frame.epi.astype(int) - 1
    if cfg["model"].get("reset_after_warmup", False):
        carry_index = carry_index - np.where(frame.epi > warmup_episodes, warmup_episodes, 0)
    frame["carry_block"] = (carry_index % int(cfg["env"]["episode_carry_blocks"])) + 1
    frame["joint_rmse"] = np.sqrt((frame.pos_rmse**2 + frame.neg_rmse**2) / 2)
    frame["type"] = typ
    frame["policy"] = policy
    numeric = frame.select_dtypes(include=["number"])
    finite = np.isfinite(numeric.to_numpy())
    expected_missing = np.zeros(finite.shape, dtype=bool)
    for column in ("neg_log_prob_mean", "log_std_mean", "log_std_min", "log_std_max"):
        if column in numeric:
            expected_missing[:, numeric.columns.get_loc(column)] = (
                (frame.epi.to_numpy() <= warmup_episodes)
                & numeric[column].isna().to_numpy()
            )
    unexplained_nonfinite = (~finite) & (~expected_missing)
    ref = cfg.get("reference", {})
    audit = {
        "name": name, "type": typ, "policy": policy,
        "wall_time": wall, "infos_episodes": len(info), "diagnostic_episodes": len(diag),
        "exploration_episodes": len(exploration), "joined_episodes": len(frame),
        "last_episode": int(frame.epi.max()),
        "all_numeric_finite_except_expected_warmup_nan": bool(not unexplained_nonfinite.any()),
        "expected_warmup_nan_values": int(expected_missing.sum()),
        "unexpected_nonfinite_values": int(unexplained_nonfinite.sum()),
        "reward_logger_agrees": reward_agrees,
        "episode_sequence_contiguous": bool(np.array_equal(frame.epi, np.arange(1, len(frame) + 1))),
        "checkpoint_files": [p.name for p in sorted(folder.glob("model_epi_*.pth"))],
        "reference_npz": ref.get("kwargs", {}).get("trajectory_npz"),
        "library_path": cfg["obs"].get("library_path"),
        "source_config": cfg.get("source_config"),
        "windows": {
            "first200_after_warmup": window_stats(frame[(frame.epi > 10) & (frame.epi <= 210)]),
            "episodes801_1000": window_stats(frame[(frame.epi >= 801) & (frame.epi <= 1000)]),
            "episodes1501_1700": window_stats(frame[(frame.epi >= 1501) & (frame.epi <= 1700)]),
            "last200": window_stats(frame.tail(200)),
            "previous200": window_stats(frame.iloc[-400:-200]),
            "last100": window_stats(frame.tail(100)),
            "previous100": window_stats(frame.iloc[-200:-100]),
        },
        "checkpoint_windows": {},
        "carry_block_last256": {},
        "operating_bounds": {},
    }
    reward_cfg = cfg["env"]["rwd_kwargs"]
    pos_max = float(reward_cfg["chamber_pos_max_kpa"])
    neg_min = float(reward_cfg["chamber_neg_min_kpa"])
    for window, values in (("all_training", frame), ("last200", frame.tail(200))):
        audit["operating_bounds"][window] = {
            "ch_pos_limit_kpa": pos_max, "ch_neg_limit_kpa": neg_min,
            "ch_pos_high_episode_ratio": float((values.ch_pos_max > pos_max).mean()),
            "ch_neg_low_episode_ratio": float((values.ch_neg_min < neg_min).mean()),
            "ch_pos_max_kpa": float(values.ch_pos_max.max()),
            "ch_neg_min_kpa": float(values.ch_neg_min.min()),
            "ch_pos_max_episode": int(values.loc[values.ch_pos_max.idxmax(), "epi"]),
            "note": "Ratios count episodes with any crossing, not time-sample violation ratios. Reward bounds are soft penalties, not pressure clipping or hard constraints.",
        }
    for episode in (500, 1000, 1500, 2000):
        audit["checkpoint_windows"][str(episode)] = window_stats(
            frame[(frame.epi > episode - 100) & (frame.epi <= episode)]
        )
    for block in range(1, 5):
        audit["carry_block_last256"][str(block)] = window_stats(frame.tail(256).query("carry_block == @block"))
    moving = frame.reward.rolling(100, min_periods=100).mean()
    best_idx = moving.idxmax()
    audit["best100_training_reward"] = {
        "last_episode": int(frame.loc[best_idx, "epi"]), "mean": float(moving.loc[best_idx]),
        "note": "Training samples include stochastic SAC actions; this is not checkpoint selection by deterministic evaluation.",
    }
    return frame, audit


def draw(ax, frame, key, policy, label=None, window=100):
    x, y = frame.epi, frame[key]
    color = COLORS[policy]
    ax.plot(x, y, color=color, alpha=0.12, lw=0.45)
    ax.plot(x, y.rolling(window, min_periods=window).mean(), color=color, lw=1.6, label=label or policy)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Episode")


def save_figures(frames, out: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, typ in enumerate((1, 2)):
        for policy in ("H", "HR"):
            f = frames[(typ, policy)]
            draw(axes[0, col], f, "reward", policy)
            draw(axes[1, col], f[f.epi >= 500], "reward", policy)
        axes[0, col].set_title(f"Type {typ}: full training return")
        axes[1, col].set_title(f"Type {typ}: episodes 500+ (100-episode moving average)")
        for ax in axes[:, col]:
            ax.set_ylabel("Episode return (higher is better)")
            ax.legend()
    fig.suptitle("SAC training return: transparent raw values + trailing mean; type-specific reference banks")
    fig.savefig(out / "training_reward.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    for col, typ in enumerate((1, 2)):
        for row, key in enumerate(("pos_rmse", "neg_rmse", "pos_bias", "neg_bias")):
            for policy in ("H", "HR"):
                draw(axes[row, col], frames[(typ, policy)], key, policy)
            axes[row, col].set_title(f"Type {typ}: {key}")
            axes[row, col].set_ylabel("kPa")
            if "bias" in key:
                axes[row, col].axhline(0, color="black", lw=0.5)
            else:
                axes[row, col].set_ylim(bottom=0, top=8)
            axes[row, col].legend()
    fig.suptitle("Recorded training tracking error; bias = reference - measured pressure")
    fig.savefig(out / "training_rmse_bias.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(14, 15), constrained_layout=True)
    for col, typ in enumerate((1, 2)):
        for row, key in enumerate(("critic_loss", "policy_loss", "alpha", "action_std_mean", "action_saturation_ratio")):
            for policy in ("H", "HR"):
                draw(axes[row, col], frames[(typ, policy)].query("epi > 10"), key, policy)
            axes[row, col].set_title(f"Type {typ}: {key}")
            if key in ("critic_loss", "alpha"):
                axes[row, col].set_yscale("log")
            axes[row, col].legend()
    fig.suptitle("SAC optimization and stochastic exploration diagnostics; policy loss is not tracking quality")
    fig.savefig(out / "training_optimization.png", dpi=180)
    plt.close(fig)

    metrics = (
        ("act_pos_sign_change_per_sec", "Positive pressure slope sign changes (not frequency in Hz)"),
        ("act_neg_sign_change_per_sec", "Negative pressure slope sign changes (not frequency in Hz)"),
        ("u_pos_in_delta", "Positive inlet mean absolute command difference"),
        ("u_pos_out_delta", "Positive outlet mean absolute command difference"),
        ("u_neg_in_delta", "Negative inlet mean absolute command difference"),
        ("u_neg_out_delta", "Negative outlet mean absolute command difference"),
    )
    fig, axes = plt.subplots(6, 2, figsize=(14, 17), constrained_layout=True)
    for col, typ in enumerate((1, 2)):
        for row, (key, title) in enumerate(metrics):
            for policy in ("H", "HR"):
                draw(axes[row, col], frames[(typ, policy)], key, policy)
            axes[row, col].set_title(f"Type {typ}: {title}", fontsize=9)
            axes[row, col].legend()
    fig.suptitle("Training jitter indicators; use deterministic time series / PSD to assess pressure oscillation")
    fig.savefig(out / "training_action_pressure_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for col, typ in enumerate((1, 2)):
        for row, key in enumerate(("reward", "pos_rmse", "neg_rmse")):
            for policy in ("H", "HR"):
                f = frames[(typ, policy)].tail(256)
                group = f.groupby("carry_block")[key]
                axes[row, col].errorbar(group.mean().index, group.mean(), yerr=group.std(), marker="o", capsize=3, color=COLORS[policy], label=policy)
            axes[row, col].set_title(f"Type {typ}: {key}, last 256 episodes")
            axes[row, col].set_xticks([1, 2, 3, 4])
            axes[row, col].set_xlabel("Carried reference block (block 1 starts after all-open reset)")
            axes[row, col].set_ylabel("Return" if key == "reward" else "kPa")
            axes[row, col].grid(alpha=0.2)
            axes[row, col].legend()
    fig.suptitle("Carry block profile: mean ± episode standard deviation (not confidence intervals)")
    fig.savefig(out / "training_carry_block.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, typ in enumerate((1, 2)):
        for row, key in enumerate(("ch_pos_max", "ch_neg_min")):
            for policy in ("H", "HR"):
                draw(axes[row, col], frames[(typ, policy)], key, policy)
            limit = 220 if row == 0 else 10
            axes[row, col].axhline(limit, color="crimson", linestyle="--", lw=1, label=f"Soft reward bound {limit} kPa")
            axes[row, col].set_title(f"Type {typ}: {'maximum chamber+ pressure' if row == 0 else 'minimum chamber- pressure'} in each episode")
            axes[row, col].set_ylabel("Absolute pressure (kPa)")
            axes[row, col].legend()
    fig.suptitle("Operating bounds in stochastic training: soft reward penalties do not guarantee constraints")
    fig.savefig(out / "training_supply_pressure.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pneu_rl/exp/lib3_physical_training_review_20261023/reward")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, audits = {}, {}
    for typ, policy in RUNS:
        frame, audit = read_run(typ, policy)
        frames[(typ, policy)] = frame
        audits[f"type{typ}_{policy}"] = audit
    comparisons = {}
    config_checks = {}
    for typ in (1, 2):
        end = min(int(frames[(typ, p)].epi.max()) for p in ("H", "HR"))
        comparisons[f"type{typ}"] = {"matched_last200_episode_end": end}
        for policy in ("H", "HR"):
            comparisons[f"type{typ}"][policy] = window_stats(frames[(typ, policy)].query("epi > @end - 200 and epi <= @end"))
        configs = [yaml.safe_load((ROOT / "pneu_rl/models" / audits[f"type{typ}_{p}"]["name"] / "cfg.yaml").read_text()) for p in ("H", "HR")]
        differences = config_differences(*configs)
        expected = ["env.include_physics_features", "env.physics_feature_mode", "source_config"]
        config_checks[f"type{typ}"] = {"different_fields": differences, "only_expected_H_HR_differences": differences == expected}
    pd.concat(frames.values(), ignore_index=True).to_csv(args.output_dir / "training_episodes.csv", index=False)
    draw_output = {"reviewed_unix": time.time(), "runs": audits, "matched_comparisons": comparisons, "config_comparability": config_checks,
                   "interpretation": ["Training return includes stochastic actions and is not deterministic checkpoint performance.", "Type-specific reference banks differ, so compare H versus HR within each type.", "Slope sign changes per second do not measure vibration amplitude or oscillation frequency."]}
    (args.output_dir / "reward_audit.json").write_text(json.dumps(draw_output, indent=2, allow_nan=False) + "\n")
    save_figures(frames, args.output_dir)
    lines = ["# 2026-10-23 SAC 학습 결과와 reward review", "", "학습 중 stochastic action이 포함된 기록이다. Deterministic quick visualization / rollout 결과와 구분한다. Type1과 type2는 reference bank가 달라 reward 숫자의 직접 비교 대상이 아니다.", "", "| Run | Episodes | Completed | Last 200 return mean ± SD | pos RMSE | neg RMSE | Final alpha |", "| --- | ---: | :---: | ---: | ---: | ---: | ---: |"]
    for key, audit in audits.items():
        tail = audit["windows"]["last200"]
        final_alpha = frames[(audit["type"], audit["policy"])].alpha.iloc[-1]
        lines.append(f"| {key} | {audit['last_episode']} | {audit['wall_time'].get('completed', False)} | {tail['reward']['mean']:.2f} ± {tail['reward']['std']:.2f} | {tail['pos_rmse']['mean']:.3f} kPa | {tail['neg_rmse']['mean']:.3f} kPa | {final_alpha:.5f} |")
    lines += ["", "같은 type 안에서 H·HR의 공통 마지막 200개 episode를 맞춰 비교하면:", ""]
    for typ in (1, 2):
        comp = comparisons[f"type{typ}"]
        h, hr = comp["H"], comp["HR"]
        pos_drop = (1 - hr["pos_rmse"]["mean"] / h["pos_rmse"]["mean"]) * 100
        neg_drop = (1 - hr["neg_rmse"]["mean"] / h["neg_rmse"]["mean"]) * 100
        lines.append(f"- Type{typ}, episode {comp['matched_last200_episode_end'] - 199}–{comp['matched_last200_episode_end']}: HR의 양압 RMSE 평균 {pos_drop:.1f}%, 음압 RMSE 평균 {neg_drop:.1f}% 감소. Return 평균은 H {h['reward']['mean']:.2f}, HR {hr['reward']['mean']:.2f}.")
    lines += ["", "학습 중 operating bound를 별도로 확인했다. 아래 비율은 해당 episode 안에서 한 번이라도 chamber+ 220 kPa를 넘은 episode 비율이며, 시간 샘플 비율과는 다르다.", "", "| Run | 전체 학습 최대 chamber+ | 마지막 200episode 최대 chamber+ | 마지막 200episode 중 >220kPa 비율 |", "| --- | ---: | ---: | ---: |"]
    for key, audit in audits.items():
        bounds = audit["operating_bounds"]
        lines.append(f"| {key} | {bounds['all_training']['ch_pos_max_kpa']:.2f} kPa (epi {bounds['all_training']['ch_pos_max_episode']}) | {bounds['last200']['ch_pos_max_kpa']:.2f} kPa | {bounds['last200']['ch_pos_high_episode_ratio'] * 100:.1f}% |")
    lines += ["", "Type1 HR은 초기 탐색 중 chamber+가 502.50 kPa까지 올라갔다. 최종 HR의 nominal 결과가 범위 안에 있더라도 전체 학습이 항상 운전 범위를 지킨 것은 아니다. Chamber bound는 soft reward penalty였으며 hard constraint나 pressure clipping이 아니다. 양의 절대압·유량 방향 등 physical feasibility와 정해 둔 운전 범위 준수는 별도 판단이다."]
    lines += ["", "Carry block 그래프를 직접 확인하면 HR의 낮은 error가 block 1뿐 아니라 네 block 전반에 유지된다. 현재 simulation 기록에서는 전체 개방 reset 직후에만 RMSE가 집중되는 패턴으로 보이지 않는다. 이는 real 초기화 mismatch가 없다는 검증은 아니다."]
    late_hr = audits["type2_HR"]["windows"]
    prev_pos, last_pos = late_hr["previous100"]["pos_rmse"]["mean"], late_hr["last100"]["pos_rmse"]["mean"]
    lines += ["", f"Type2 HR에는 종료 직전 작은 양압 성능 저하가 남는다. 직전 100episode 대비 마지막 100episode의 양압 RMSE 평균은 {prev_pos:.3f} → {last_pos:.3f} kPa로 증가했다. 전체 학습 붕괴로 볼 정도는 아니지만 마지막 checkpoint가 모든 채널에서 가장 좋다는 뜻은 아니다. Checkpoint 선택에는 같은 reference의 deterministic evaluation을 사용해야 한다."]
    lines += ["", "- 학습 중간에는 일시적인 reward regression이 있었다. Type1 H는 약 1200episode, type2 HR은 약 1000episode, type2 H는 약 1600episode 주변에서 저하 후 회복했다. 최종 구간의 지속적인 loss divergence로 보이지는 않는다. Regression의 원인을 특정한 결과는 아니다.", "- 같은 type의 H·HR config는 physics feature 포함 여부·mode·원본 config 경로만 다르다. Reward, reference, simulator, seed, SAC hyperparameters는 동일하다.", "- 단일 seed의 학습 결과이므로 observation 구성의 일반적 우월성이나 real 성능을 입증한 결과는 아니다.", "- 모든 기록의 NaN/Inf, episode 연속성, logger 간 reward 일치 여부는 reward_audit.json에 저장했다. 처음 10개 fixed-action warmup episode에는 log-probability / log-std 값이 정의되지 않아 NaN인 40개 셀이 run마다 있으며, 예외로 구분했다.", "- RMSE 표는 episode RMSE의 평균이다. 전체 시계열을 합친 global RMSE와는 다르다.", "- Reward는 클수록 좋다. Policy loss 값 자체는 추종 성능 지표가 아니다.", "- Action saturation은 raw policy command ≤0.01 또는 ≥0.99인 비율이다. Physical command ≤0.802 또는 ≥0.998에 해당하며, 유량 포화와 동일한 의미는 아니다.", "- action_std_mean은 episode 중 action의 표준편차 평균이라 reference 변화에 대한 action 변화도 포함한다. 순수 exploration noise 크기로 해석하지 않는다.", "- Pressure slope sign-change rate는 진동 주파수(Hz)가 아니다. 작은 stochastic ripple도 집계하므로 진동 판단에는 deterministic waveform, amplitude, PSD가 필요하다.", "- Valve command의 평균 곱(conflict)은 physical command가 0.8–1.0이라 큰 값이 기본적으로 나오며 개방면적/유량 충돌의 증거로 해석하지 않았다.", "- Carry block 번호는 10episode warmup 후 강제 hard reset이 있다는 사실을 반영했다. 따라서 학습 episode 11·15·19…가 block 1이다.", "", "![Reward](training_reward.png)", "", "![RMSE and bias](training_rmse_bias.png)", "", "![Optimization](training_optimization.png)", "", "![Jitter diagnostics](training_action_pressure_diagnostics.png)", "", "![Carry block profile](training_carry_block.png)"]
    lines += ["", "![Supply pressure and operating bounds](training_supply_pressure.png)"]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: {"episode": a["last_episode"], "completed": a["wall_time"].get("completed", False), "finite_except_expected_warmup": a["all_numeric_finite_except_expected_warmup_nan"], "last200_reward": a["windows"]["last200"]["reward"]["mean"], "pos_rmse": a["windows"]["last200"]["pos_rmse"]["mean"], "neg_rmse": a["windows"]["last200"]["neg_rmse"]["mean"]} for k, a in audits.items()}, indent=2))


if __name__ == "__main__":
    main()
