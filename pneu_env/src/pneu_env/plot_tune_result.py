#!/usr/bin/env python3

import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import optimizer


CONST_TO_FUNC = {
    "CHAMBER_POS_PARAMS": "chamber_pos",
    "CHAMBER_NEG_PARAMS": "chamber_neg",
    "ACT_POS_IN_PARAMS": "act_pos_in",
    "ACT_POS_OUT_PARAMS": "act_pos_out",
    "ACT_NEG_IN_PARAMS": "act_neg_in",
    "ACT_NEG_OUT_PARAMS": "act_neg_out",
}


def extract_numbers(text):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]


def parse_const_param_blocks(text):
    params_by_func = {}
    pattern = re.compile(
        r"const\s+ValveModelParams\s+([A-Za-z0-9_]+)\s*=\s*\{(.*?)\}\s*;",
        re.DOTALL,
    )
    for const_name, body in pattern.findall(text):
        func_name = CONST_TO_FUNC.get(const_name)
        if func_name is None:
            continue
        values = extract_numbers(body)
        if len(values) != 13:
            raise ValueError(f"{const_name} 파라미터 개수가 13개가 아닙니다: {len(values)}")
        params_by_func[func_name] = np.asarray(values, dtype=np.float64)
    return params_by_func


def parse_make_params_blocks(text):
    params_by_func = {}
    pattern = re.compile(
        r"ValveModelParams\s+([A-Za-z0-9_]+)\s*\(\s*\)\s*\{(.*?)\}",
        re.DOTALL,
    )
    for func_name, body in pattern.findall(text):
        if "make_params" not in body:
            continue
        values = extract_numbers(body)
        if len(values) == 14:
            values = values[1:]
        if len(values) != 13:
            raise ValueError(f"{func_name} 파라미터 개수가 13개가 아닙니다: {len(values)}")
        params_by_func[func_name] = np.asarray(values, dtype=np.float64)
    return params_by_func


def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    params_by_func = parse_const_param_blocks(text)
    params_by_func.update(parse_make_params_blocks(text))
    if not params_by_func:
        raise ValueError(f"파라미터 블록을 찾지 못했습니다: {path}")
    return params_by_func


def next_available_image_path(path):
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".png"
        path = root + ext
    idx = 0
    while os.path.exists(path):
        idx += 1
        path = f"{root}_{idx}{ext}"
    return path


def default_output_path(params_path):
    directory = os.path.dirname(os.path.abspath(params_path)) or os.getcwd()
    stem = os.path.splitext(os.path.basename(params_path))[0]
    return os.path.join(directory, f"{stem}_plot.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="optimizer에 넣었던 데이터 CSV 경로")
    parser.add_argument("params", help="optimizer 결과 txt 경로")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--valves", default=None, help="그릴 밸브 인덱스. 생략하면 txt에 있는 밸브만 그림")
    parser.add_argument("--output", default=None, help="저장할 이미지 경로. 생략하면 params 파일 옆에 *_plot.png 저장")
    parser.add_argument("--no-show", action="store_true", help="plot window를 띄우지 않고 저장만 함")
    args = parser.parse_args()
    args.data = args.data.strip()
    args.params = args.params.strip()
    if args.output is not None:
        args.output = args.output.strip()

    if args.window_size < 1:
        raise ValueError("--window-size 는 1 이상의 정수여야 합니다.")

    params_by_func = load_params(args.params)
    selected_cfgs = [cfg for cfg in optimizer.CONFIGS if cfg["func_name"] in params_by_func]
    if args.valves is not None:
        selected = optimizer.parse_valve_selection(args.valves, max_valve=max(cfg["idx"] for cfg in optimizer.CONFIGS))
        selected_cfgs = [cfg for cfg in selected_cfgs if cfg["idx"] in selected]
    if not selected_cfgs:
        raise ValueError("선택 조건에 맞는 파라미터/밸브가 없습니다.")

    n_plots = len(selected_cfgs)
    ncols = 1 if n_plots <= 3 else 2
    nrows = int(math.ceil(n_plots / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.5 * ncols, 4.8 * nrows),
        sharex=False,
        facecolor="w",
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.canvas.manager.set_window_title("Valve Fit Replay")
    fig.suptitle(
        f"Valve Fit Replay | params={os.path.basename(args.params)} | "
        f"window_size={args.window_size} | valves={[cfg['idx'] for cfg in selected_cfgs]}",
        fontsize=14,
    )

    for i, cfg in enumerate(selected_cfgs):
        data = optimizer.load_and_preprocess(
            args.data,
            cfg,
            start_time_sec=args.start,
            window_size=args.window_size,
        )
        params = params_by_func[cfg["func_name"]]
        _, q_pred = optimizer.simulate_physics_model(data, params)
        rmse = float(np.sqrt(np.mean((data["Q"] - q_pred) ** 2)))
        r_sq = optimizer.compute_r2(data["Q"], q_pred)
        hf_ratio = optimizer.high_freq_ratio(data["Q"], q_pred)
        t_plot = data["Time"] - args.start

        ax = axes[i]
        ax.plot(t_plot, data["Q"], "k-", linewidth=3, label="Actual Q")
        ax.plot(t_plot, q_pred, "r--", linewidth=2.2, label="Fitted Q")
        ax.set_title(
            f"[{cfg['idx']}] {cfg['name']} | window={args.window_size}\n"
            f"RMSE={rmse:.4f} | R²={r_sq * 100:.1f}% | HF ratio={hf_ratio:.3f}",
            fontsize=12,
        )
        ax.set_ylabel("Flow", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True)

    for idx_ax in range(n_plots):
        row_idx = idx_ax // ncols
        if row_idx == nrows - 1:
            axes[idx_ax].set_xlabel("Time [s]", fontsize=12)

    output_path = args.output if args.output is not None else default_output_path(args.params)
    output_path = next_available_image_path(output_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"[INFO] replay image 저장 완료: {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
