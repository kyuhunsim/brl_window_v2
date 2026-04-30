import argparse
import ctypes
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
from utils.utils import setup_plot_style

setup_plot_style({"legend.fontsize": 18})


# C++ 공유 라이브러리 로드
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuner/sim_core.so')
if not os.path.exists(lib_path):
    raise FileNotFoundError("sim_core.so 라이브러리가 없습니다. 터미널에서 'g++ -O3 -shared -fPIC -o sim_core.so sim_core.cpp' 를 실행하세요.")

sim_lib = ctypes.CDLL(lib_path)
sim_lib.simulate_core.restype = ctypes.c_double
sim_lib.simulate_core.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double)
]

CONFIGS = [
    {
        'idx': 1,
        'name': 'ctrl1 (Pchamber pos -> ATM)',
        'u_col': 'ctrl1',
        'Q_col': 'flow1',
        'P_in_col': 'press_pos',
        'P_out_col': 'ATM',
        'base': [0.177485, 24.9354, 0.0918, 0.000251, 0.00000, 363318.0739, 1.6334, 0.1516, 83.5718, 2.3474, 0.9792, 3.6058, 1.5719],
        'func_name': 'chamber_pos',
        'flow_multiplier_token': 'POS_VALVE_NUM',
        'output_order': 1,
    },
    {
        'idx': 2,
        'name': 'ctrl2 (ATM -> Pchamber neg)',
        'u_col': 'ctrl2',
        'Q_col': 'flow2',
        'P_in_col': 'ATM',
        'P_out_col': 'press_neg',
        'base': [0.252364, 49.2420, 0.0821, 0.000124, -0.00002, 77568.1783, 753.6405, 0.1452, 11752.9849, 4.5513, 2.0892, 2.4077, 0.7968],
        'func_name': 'chamber_neg',
        'flow_multiplier_token': 'NEG_VALVE_NUM',
        'output_order': 2,
    },
    {
        'idx': 3,
        'name': 'ctrl3 (Pchamber pos -> Pactuator pos)',
        'u_col': 'ctrl3',
        'Q_col': 'flow3',
        'P_in_col': 'press_pos',
        'P_out_col': 'act_pos_press',
        'base': [0.228231, 37.4621, 0.0278, 0.000114, 0.00000, 129581.8357, 6.7852, 0.1294, 17385.9804, 3.8342, 2.5870, 5.3290, 2.3735],
        'func_name': 'act_pos_in',
        'flow_multiplier_token': '1.0',
        'output_order': 3,
    },
    {
        'idx': 4,
        'name': 'ctrl4 (Pactuator pos -> ATM)',
        'u_col': 'ctrl4',
        'Q_col': 'flow4',
        'P_in_col': 'act_pos_press',
        'P_out_col': 'ATM',
        'base': [0.147350, 29.8726, 0.0514, 0.000236, 0.00001, 388775.9147, 2.5299, 0.0932, 431.3010, 11.7718, 5.0262, 2.0747, 0.7610],
        'func_name': 'act_pos_out',
        'flow_multiplier_token': '1.0',
        'output_order': 4,
    },
    {
        'idx': 5,
        'name': 'ctrl5 (ATM -> Pactuator neg)',
        'u_col': 'ctrl5',
        'Q_col': 'flow5',
        'P_in_col': 'ATM',
        'P_out_col': 'act_neg_press',
        'base': [0.216261, 60.6026, 0.1120, 0.000054, 0.00000, 151610.7046, 1.5688, 0.8781, 14059.9816, 1.2952, 0.7522, 25.2916, 12.0013],
        'func_name': 'act_neg_in',
        'flow_multiplier_token': '1.0',
        'output_order': 5,
    },
    {
        'idx': 6,
        'name': 'ctrl6 (Pactuator neg -> Pchamber neg)',
        'u_col': 'ctrl6',
        'Q_col': 'flow6',
        'P_in_col': 'act_neg_press',
        'P_out_col': 'press_neg',
        'base': [0.191616, 43.4833, 0.0000, 0.000520, 0.00063, 1079.1638, 0.8231, 2.1481, 109303.4554, 11.6611, 2.7509, 10.7609, 2.5621],
        'func_name': 'act_neg_out',
        'flow_multiplier_token': '1.0',
        'output_order': 6,
    }
]


def get_phi(P_in, P_out, kappa):
    Pr = np.clip(P_out / P_in, 0, 1.0)
    P_cr = (2 / (kappa + 1)) ** (kappa / (kappa - 1))
    phi = np.zeros_like(Pr)
    idx_choked = Pr <= P_cr
    phi[idx_choked] = np.sqrt(kappa * (2 / (kappa + 1)) ** ((kappa + 1) / (kappa - 1)))
    idx_sub = (Pr > P_cr) & (Pr <= 1)
    phi[idx_sub] = np.sqrt((2 * kappa) / (kappa - 1)) * np.sqrt(
        Pr[idx_sub] ** (2 / kappa) - Pr[idx_sub] ** ((kappa + 1) / kappa)
    )
    return phi


def maybe_smooth(arr, window_size):
    if window_size <= 1:
        return np.asarray(arr, dtype=np.float64)
    return np.asarray(uniform_filter1d(arr, size=window_size), dtype=np.float64)


def parse_valve_selection(text, max_valve=6):
    s = str(text).strip().lower()
    if s in ("all", "*"):
        return set(range(1, max_valve + 1))

    selected = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2:
                raise ValueError(f"잘못된 밸브 범위 토큰: {token}")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    invalid = sorted(v for v in selected if v < 1 or v > max_valve)
    if invalid:
        raise ValueError(f"밸브 인덱스 범위 초과: {invalid} (허용: 1~{max_valve})")
    if not selected:
        raise ValueError("선택된 밸브가 없습니다.")
    return selected


def sanitize_param_vector(params, wn_min=0.0, wn_max=150.0, zeta_min=0.0, zeta_max=30.0):
    p = np.asarray(params, dtype=np.float64).copy()
    if p.shape[0] != 13:
        raise ValueError(f"파라미터 길이가 13이 아닙니다: {p.shape[0]}")
    wn_lo = min(float(wn_min), float(wn_max))
    wn_hi = max(float(wn_min), float(wn_max))
    zeta_lo = min(float(zeta_min), float(zeta_max))
    zeta_hi = max(float(zeta_min), float(zeta_max))
    p[0] = abs(p[0])            # a_max
    p[1] = abs(p[1])            # k_shape
    p[2] = abs(p[2])            # c_k
    # p[3], p[4]는 부호 허용(c_p, c_z)
    p[5] = abs(p[5])            # a_bw
    p[6] = abs(p[6])            # beta_bw
    p[7] = abs(p[7])            # gamma_bw
    p[8] = abs(p[8])            # alpha_shape
    p[9] = np.clip(abs(p[9]), wn_lo, wn_hi)       # wn_up
    p[10] = np.clip(abs(p[10]), zeta_lo, zeta_hi) # zeta_up
    p[11] = np.clip(abs(p[11]), wn_lo, wn_hi)     # wn_down
    p[12] = np.clip(abs(p[12]), zeta_lo, zeta_hi) # zeta_down
    return p


def high_freq_ratio(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0

    # 최적화 중 비정상 수치(inf/nan/초대형)가 생길 수 있어 안전하게 제한
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e9, neginf=-1e9)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    y_true = np.clip(y_true, -1e9, 1e9)
    y_pred = np.clip(y_pred, -1e9, 1e9)

    hf_true = float(np.std(np.diff(y_true), dtype=np.float64))
    hf_pred = float(np.std(np.diff(y_pred), dtype=np.float64))
    return hf_pred / max(hf_true, eps)


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e9, neginf=-1e9)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (ss_res / ss_tot)


def simulate_physics_model(data, params):
    N = len(data['I'])
    Q_pred = np.zeros(N, dtype=np.float64)

    error = sim_lib.simulate_core(
        N,
        data['I'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        data['State'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        data['P_in_abs'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        data['Phi'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        data['Q'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        data['dt'],
        params[0], params[1], params[2], params[3], params[4],
        params[5], params[6], params[7], params[8],
        params[9], params[10], params[11], params[12],
        Q_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return error, Q_pred


def compute_global_error(
    params,
    all_data,
    hf_weight=0.0,
    hf_target=1.0,
    r2_floor=-1.0,
    r2_weight=0.0,
    wn_min=0.0,
    wn_max=150.0,
    zeta_min=0.0,
    zeta_max=30.0,
):
    params_eval = sanitize_param_vector(
        params,
        wn_min=wn_min,
        wn_max=wn_max,
        zeta_min=zeta_min,
        zeta_max=zeta_max,
    )
    total_error = 0.0
    for data in all_data:
        base_error, q_pred = simulate_physics_model(data, params_eval)
        if not np.isfinite(base_error) or np.any(~np.isfinite(q_pred)):
            return 1e30
        total_error += float(base_error)
        if hf_weight > 0.0:
            hf_ratio = high_freq_ratio(data['Q'], q_pred)
            hf_excess = max(0.0, hf_ratio - float(hf_target))
            # base_error(대체로 SSE 스케일)에 맞추기 위해 샘플 수를 곱해 penalty 스케일 정규화
            total_error += float(hf_weight) * len(q_pred) * (hf_excess ** 2)
        if r2_weight > 0.0 and r2_floor > 0.0:
            r2_val = compute_r2(data['Q'], q_pred)
            r2_deficit = max(0.0, float(r2_floor) - r2_val)
            total_error += float(r2_weight) * len(q_pred) * (r2_deficit ** 2)
    return float(total_error)


def load_and_preprocess(
    csv_path,
    config,
    start_time_sec=100.0,
    end_time_sec=None,
    tail_cut_sec=0.0,
    drop_incomplete_last_command=False,
    settle_sec=2.0,
    flow_delay_samples=0,
    flow_delay_sec=None,
    I_MAX=0.30,
    window_size=1,
    input_window_size=None,
    flow_window_size=None,
):
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns:
        if 'curr_time' in df.columns:
            df = df.rename(columns={'curr_time': 'time'})
        else:
            raise ValueError("CSV에 time 또는 curr_time 컬럼이 필요합니다.")
    valid_idx = df['time'] >= start_time_sec
    if not valid_idx.any():
        valid_idx = df['time'] >= 0
    df = df[valid_idx].reset_index(drop=True)
    if end_time_sec is not None:
        df = df[df['time'] <= float(end_time_sec)].reset_index(drop=True)
    if tail_cut_sec and float(tail_cut_sec) > 0.0:
        tail_end = float(df['time'].iloc[-1]) - float(tail_cut_sec)
        df = df[df['time'] <= tail_end].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(
            f"유효 데이터가 너무 적습니다: {len(df)} rows "
            f"(start={start_time_sec}, end={end_time_sec}, tail_cut={tail_cut_sec})"
        )
    if drop_incomplete_last_command:
        u_for_segment = df[config['u_col']].to_numpy(dtype=np.float64)
        t_for_segment = df['time'].to_numpy(dtype=np.float64)
        change_idx = np.flatnonzero(np.abs(np.diff(u_for_segment)) > 1e-9) + 1
        if change_idx.size > 0:
            last_idx = int(change_idx[-1])
            remaining = float(t_for_segment[-1] - t_for_segment[last_idx])
            if remaining < float(settle_sec):
                df = df.iloc[:last_idx].reset_index(drop=True)
                if len(df) < 2:
                    raise ValueError(
                        f"마지막 incomplete command 제거 후 데이터가 너무 적습니다: {len(df)} rows "
                        f"(valve={config['idx']}, settle_sec={settle_sec})"
                    )

    flow_delay_samples = int(flow_delay_samples)
    if flow_delay_sec is not None:
        dt_median = float(np.median(np.diff(df['time'].to_numpy(dtype=np.float64))))
        if dt_median <= 0.0:
            raise ValueError(f"flow-delay-sec 변환 실패: median dt={dt_median}")
        flow_delay_samples += int(round(float(flow_delay_sec) / dt_median))
    effective_flow_delay_samples = flow_delay_samples
    q_values = df[config['Q_col']].to_numpy(dtype=np.float64)
    if flow_delay_samples > 0:
        if len(df) <= flow_delay_samples + 1:
            raise ValueError(
                f"flow delay 적용 후 데이터가 너무 적습니다: rows={len(df)}, "
                f"delay_samples={flow_delay_samples}"
            )
        # Measured flow is assumed to lag command/pressure by delay_samples.
        # Pair command/pressure at k with measured flow at k + delay.
        q_values = q_values[flow_delay_samples:]
        df = df.iloc[:-flow_delay_samples].reset_index(drop=True)
    elif flow_delay_samples < 0:
        lead_samples = abs(flow_delay_samples)
        if len(df) <= lead_samples + 1:
            raise ValueError(
                f"flow delay 적용 후 데이터가 너무 적습니다: rows={len(df)}, "
                f"delay_samples={flow_delay_samples}"
            )
        q_values = q_values[:-lead_samples]
        df = df.iloc[lead_samples:].reset_index(drop=True)

    data = {'name': config['name']}
    data['flow_delay_samples'] = effective_flow_delay_samples
    data['Time'] = df['time'].values
    data['dt'] = np.mean(np.diff(data['Time']))
    input_window = int(window_size if input_window_size is None else input_window_size)
    flow_window = int(window_size if flow_window_size is None else flow_window_size)
    data['input_window_size'] = input_window
    data['flow_window_size'] = flow_window

    u_raw = df[config['u_col']].values
    data['u'] = np.clip((u_raw - 0.5) * 2.0, 0.0, 1.0)
    data['I'] = np.ascontiguousarray(maybe_smooth(data['u'] * I_MAX, input_window), dtype=np.float64)

    P_in = np.full(len(df), 101.325) if config['P_in_col'] == 'ATM' else df[config['P_in_col']].values
    P_out = np.full(len(df), 101.325) if config['P_out_col'] == 'ATM' else df[config['P_out_col']].values

    data['P_in_abs'] = np.ascontiguousarray(P_in, dtype=np.float64)
    data['P_out_abs'] = np.ascontiguousarray(P_out, dtype=np.float64)
    data['Q'] = np.ascontiguousarray(maybe_smooth(q_values, flow_window), dtype=np.float64)
    data['Phi'] = np.ascontiguousarray(get_phi(data['P_in_abs'], data['P_out_abs'], 1.4), dtype=np.float64)

    N = len(data['I'])
    State = np.zeros(N, dtype=np.float64)
    for k in range(1, N):
        if data['I'][k] > data['I'][k - 1] + 1e-4:
            State[k] = 1.0
        elif data['I'][k] < data['I'][k - 1] - 1e-4:
            State[k] = 0.0
        else:
            State[k] = State[k - 1]
    State[0] = State[1] if N > 1 else 0.0
    data['State'] = np.ascontiguousarray(State, dtype=np.float64)

    return data


def sanitize_params(opt_p):
    return {
        'a_max': abs(opt_p[0]),
        'k_shape': abs(opt_p[1]),
        'c_k': abs(opt_p[2]),
        'c_p': opt_p[3],
        'c_z': opt_p[4],
        'a_bw': abs(opt_p[5]),
        'beta_bw': abs(opt_p[6]),
        'gamma_bw': abs(opt_p[7]),
        'alpha_shape': abs(opt_p[8]),
        'wn_up': min(abs(opt_p[9]), 150),
        'zeta_up': abs(opt_p[10]),
        'wn_down': min(abs(opt_p[11]), 150),
        'zeta_down': abs(opt_p[12]),
    }


def format_param_value(value):
    return f"{float(value):.12g}"


def format_cpp_block(cfg, params_dict):
    const_names = {
        'chamber_pos': 'CHAMBER_POS_PARAMS',
        'chamber_neg': 'CHAMBER_NEG_PARAMS',
        'act_pos_in': 'ACT_POS_IN_PARAMS',
        'act_pos_out': 'ACT_POS_OUT_PARAMS',
        'act_neg_in': 'ACT_NEG_IN_PARAMS',
        'act_neg_out': 'ACT_NEG_OUT_PARAMS',
    }
    const_name = const_names.get(cfg['func_name'], f"{cfg['func_name'].upper()}_PARAMS")
    lines = [
        f"const ValveModelParams {const_name} = {{",
        f"    {format_param_value(params_dict['a_max'])}, {format_param_value(params_dict['k_shape'])}, "
        f"{format_param_value(params_dict['c_k'])}, {format_param_value(params_dict['c_p'])}, "
        f"{format_param_value(params_dict['c_z'])},",
        f"    {format_param_value(params_dict['a_bw'])}, {format_param_value(params_dict['beta_bw'])}, "
        f"{format_param_value(params_dict['gamma_bw'])}, {format_param_value(params_dict['alpha_shape'])},",
        f"    {format_param_value(params_dict['wn_up'])}, {format_param_value(params_dict['zeta_up'])}, "
        f"{format_param_value(params_dict['wn_down'])}, {format_param_value(params_dict['zeta_down'])}",
        "};",
    ]
    return "\n".join(lines)


def write_cpp_output(results, output_path, header_text=None):
    ordered = sorted(results, key=lambda x: x['cfg']['output_order'])
    blocks = [format_cpp_block(item['cfg'], item['params_dict']) for item in ordered]
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if header_text:
            f.write(header_text.rstrip() + "\n\n")
        f.write("\n\n\n".join(blocks) + "\n")


def build_output_header(args, selected_valves, results):
    input_window = args.input_window_size if args.input_window_size is not None else args.window_size
    flow_window = args.flow_window_size if args.flow_window_size is not None else args.window_size
    lines = [
        "// Optimizer metadata",
        f"// data: {args.data}",
        f"// valves: {sorted(selected_valves)}",
        f"// start: {args.start}",
        f"// end: {args.end if args.end is not None else 'file_end'}",
        f"// tail_cut: {args.tail_cut}",
        f"// drop_incomplete_last_command: {args.drop_incomplete_last_command}",
        f"// settle_sec: {args.settle_sec}",
        f"// window_size_compat: {args.window_size}",
        f"// input_window_size: {input_window}",
        f"// flow_window_size: {flow_window}",
        f"// flow_delay_samples_arg: {args.flow_delay_samples}",
        f"// flow_delay_sec_arg: {args.flow_delay_sec if args.flow_delay_sec is not None else 'none'}",
        f"// tune_mode: {args.tune_mode}",
        f"// dynamic_valves: {args.dynamic_valves}",
        f"// samples: {args.samples}",
        f"// seed: {args.seed}",
        f"// hf_weight: {args.hf_weight}",
        f"// hf_target: {args.hf_target}",
        f"// hf_valves: {args.hf_valves}",
        f"// r2_floor: {args.r2_floor}",
        f"// r2_weight: {args.r2_weight}",
        f"// wn_range: [{args.wn_min}, {args.wn_max}]",
        f"// zeta_range: [{args.zeta_min}, {args.zeta_max}]",
    ]
    for item in sorted(results, key=lambda x: x['cfg']['output_order']):
        lines.append(
            f"// valve{item['cfg']['idx']}: best_error={item['best_error']:.12g}, "
            f"effective_flow_delay_samples={item.get('flow_delay_samples', 'unknown')}, "
            f"input_window_size={item.get('input_window_size', input_window)}, "
            f"flow_window_size={item.get('flow_window_size', flow_window)}"
        )
    return "\n".join(lines)


def resolve_result_paths(output_name, result_dir="tune_result"):
    raw_name = str(output_name).strip() if output_name is not None else ""
    if not raw_name:
        raw_name = "output.txt"

    base_name = os.path.basename(raw_name)
    stem, _ = os.path.splitext(base_name)
    if not stem:
        stem = "output"

    os.makedirs(result_dir, exist_ok=True)
    idx = 0
    while True:
        suffix = "" if idx == 0 else f"_{idx}"
        txt_path = os.path.join(result_dir, f"{stem}{suffix}.txt")
        image_path = os.path.join(result_dir, f"{stem}{suffix}.png")
        if not os.path.exists(txt_path) and not os.path.exists(image_path):
            return txt_path, image_path
        idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="데이터 CSV 파일 경로")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None, help="사용할 마지막 절대 time [s]. 생략하면 파일 끝까지 사용")
    parser.add_argument("--tail-cut", type=float, default=0.0, help="필터링 후 마지막 N초를 버림")
    parser.add_argument(
        "--drop-incomplete-last-command",
        action="store_true",
        help="마지막 ctrl 변화 이후 settle-sec 만큼 데이터가 없으면 해당 마지막 command 구간을 버림",
    )
    parser.add_argument("--settle-sec", type=float, default=2.0, help="마지막 command 완료 판단에 필요한 최소 후속 시간 [s]")
    parser.add_argument(
        "--flow-delay-samples",
        type=int,
        default=0,
        help="실측 flow가 ctrl/pressure보다 늦게 기록된 sample 수. 양수이면 Q[k+delay]를 ctrl[k]와 비교",
    )
    parser.add_argument(
        "--flow-delay-sec",
        type=float,
        default=None,
        help="실측 flow 지연 [s]. median dt로 sample 수를 계산해 --flow-delay-samples에 더함",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="랜덤 시작점 seed. 음수이면 매 실행마다 랜덤")
    parser.add_argument("--window-size", type=int, default=1, help="호환용 공통 moving average window size (개별 옵션 생략 시 사용)")
    parser.add_argument("--input-window-size", type=int, default=None, help="입력 I에만 적용할 moving average window size")
    parser.add_argument("--flow-window-size", type=int, default=None, help="실측 유량 Q에만 적용할 moving average window size")
    parser.add_argument("--output", default="output.txt", help="tune_result에 저장할 결과 파일 이름(.txt/.png는 같은 이름으로 저장)")
    parser.add_argument("--result-dir", default="tune_result", help="txt와 image 결과를 저장할 폴더")
    parser.add_argument("--valves", default="all", help="최적화할 밸브 인덱스. 예) all, 3-6, 3,4,5,6")
    parser.add_argument(
        "--tune-mode",
        choices=["all", "dynamic"],
        default="all",
        help="all: 13개 파라미터 전체 최적화, dynamic: wn/zeta(4개)만 최적화",
    )
    parser.add_argument(
        "--dynamic-valves",
        default="3-6",
        help="tune-mode=dynamic를 적용할 밸브 인덱스(부분 적용 가능). 예) 3-6",
    )
    parser.add_argument(
        "--hf-weight",
        type=float,
        default=0.0,
        help="HF penalty 가중치(0이면 비활성). 권장 시작값: 0.5~2.0",
    )
    parser.add_argument(
        "--hf-target",
        type=float,
        default=1.0,
        help="허용 HF ratio 목표치(std(diff(pred))/std(diff(real))). 이 값 초과분만 penalty 적용",
    )
    parser.add_argument(
        "--hf-valves",
        default="3-6",
        help="HF penalty 적용 대상 밸브 인덱스. 예) 3-6",
    )
    parser.add_argument("--wn-min", type=float, default=0.2, help="wn 하한(동특성 파라미터 안정화용)")
    parser.add_argument("--wn-max", type=float, default=120.0, help="wn 상한(동특성 파라미터 안정화용)")
    parser.add_argument("--zeta-min", type=float, default=0.05, help="zeta 하한(동특성 파라미터 안정화용)")
    parser.add_argument("--zeta-max", type=float, default=30.0, help="zeta 상한(동특성 파라미터 안정화용)")
    parser.add_argument(
        "--r2-floor",
        type=float,
        default=0.0,
        help="R² 최소 기준(0 이하이면 비활성). 기준 미달 시 penalty 적용",
    )
    parser.add_argument(
        "--r2-weight",
        type=float,
        default=0.0,
        help="R² floor penalty 가중치(0이면 비활성). 권장 시작값: 0.5~3.0",
    )
    args = parser.parse_args()

    if args.window_size < 1:
        raise ValueError("--window-size 는 1 이상의 정수여야 합니다.")
    if args.input_window_size is not None and args.input_window_size < 1:
        raise ValueError("--input-window-size 는 1 이상의 정수여야 합니다.")
    if args.flow_window_size is not None and args.flow_window_size < 1:
        raise ValueError("--flow-window-size 는 1 이상의 정수여야 합니다.")
    if args.end is not None and args.end <= args.start:
        raise ValueError("--end 는 --start 보다 커야 합니다.")
    if args.tail_cut < 0:
        raise ValueError("--tail-cut 은 0 이상의 값이어야 합니다.")
    if args.settle_sec < 0:
        raise ValueError("--settle-sec 은 0 이상의 값이어야 합니다.")
    if args.samples < 1:
        raise ValueError("--samples 는 1 이상의 정수여야 합니다.")
    if args.hf_weight < 0:
        raise ValueError("--hf-weight 는 0 이상의 값이어야 합니다.")
    if args.hf_target <= 0:
        raise ValueError("--hf-target 는 0보다 커야 합니다.")
    if args.r2_weight < 0:
        raise ValueError("--r2-weight 는 0 이상의 값이어야 합니다.")
    if args.r2_floor >= 1.0:
        raise ValueError("--r2-floor 는 1.0 미만이어야 합니다.")
    if args.wn_min <= 0 or args.wn_max <= 0:
        raise ValueError("--wn-min/--wn-max 는 0보다 커야 합니다.")
    if args.zeta_min < 0 or args.zeta_max <= 0:
        raise ValueError("--zeta-min/--zeta-max 범위를 확인하세요.")
    if args.seed >= 0:
        np.random.seed(args.seed)

    max_valve_idx = max(cfg["idx"] for cfg in CONFIGS)
    selected_valves = parse_valve_selection(args.valves, max_valve=max_valve_idx)
    dynamic_valves = parse_valve_selection(args.dynamic_valves, max_valve=max_valve_idx)
    hf_valves = parse_valve_selection(args.hf_valves, max_valve=max_valve_idx)
    selected_cfgs = [cfg for cfg in CONFIGS if cfg["idx"] in selected_valves]
    if not selected_cfgs:
        raise ValueError("선택 조건에 맞는 밸브가 없습니다.")

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

    fig.canvas.manager.set_window_title("Valve Optimization Fit")
    fig.suptitle(
        f"Valve Fit Results | input_window={args.input_window_size if args.input_window_size is not None else args.window_size} | "
        f"flow_window={args.flow_window_size if args.flow_window_size is not None else args.window_size} | "
        f"valves={sorted(selected_valves)} | "
        f"tune_mode={args.tune_mode} | hf_weight={args.hf_weight} | "
        f"start={args.start} | end={args.end if args.end is not None else 'file'} | tail_cut={args.tail_cut} | "
        f"drop_last={args.drop_incomplete_last_command}({args.settle_sec}s) | "
        f"flow_delay={args.flow_delay_samples} samples"
        f"{'' if args.flow_delay_sec is None else f' + {args.flow_delay_sec}s'}",
        fontsize=14,
    )

    optimization_results = []

    for i, cfg in enumerate(selected_cfgs):
        print(f"\n=======================================================")
        print(f" ▶ [{cfg['idx']}번] 최적화 시작 : {cfg['name']} (C++ 가속 적용됨)")
        print(f"=======================================================")
        print(f"   - input_window_size = {args.input_window_size if args.input_window_size is not None else args.window_size}")
        print(f"   - flow_window_size = {args.flow_window_size if args.flow_window_size is not None else args.window_size}")
        tune_dynamic = (args.tune_mode == "dynamic") and (cfg["idx"] in dynamic_valves)
        hf_weight_local = args.hf_weight if cfg["idx"] in hf_valves else 0.0
        r2_weight_local = args.r2_weight if cfg["idx"] in hf_valves else 0.0
        print(f"   - tune_mode = {'dynamic(wn/zeta only)' if tune_dynamic else 'all(13 params)'}")
        print(
            f"   - hf_penalty = {hf_weight_local:.4f} "
            f"(target={args.hf_target:.3f}, 적용대상={'예' if cfg['idx'] in hf_valves else '아니오'})"
        )
        print(
            f"   - r2_floor_penalty = {r2_weight_local:.4f} "
            f"(floor={args.r2_floor:.3f}, 적용대상={'예' if cfg['idx'] in hf_valves else '아니오'})"
        )
        data = load_and_preprocess(
            args.data,
            cfg,
            start_time_sec=args.start,
            end_time_sec=args.end,
            tail_cut_sec=args.tail_cut,
            drop_incomplete_last_command=args.drop_incomplete_last_command,
            settle_sec=args.settle_sec,
            flow_delay_samples=args.flow_delay_samples,
            flow_delay_sec=args.flow_delay_sec,
            window_size=args.window_size,
            input_window_size=args.input_window_size,
            flow_window_size=args.flow_window_size,
        )
        print(f"   - effective_flow_delay_samples = {data['flow_delay_samples']}")
        all_data = [data]
        base_initial = np.array(cfg['base'])
        dyn_indices = np.array([9, 10, 11, 12], dtype=np.int64)

        def objective_full(x):
            return compute_global_error(
                x,
                all_data,
                hf_weight=hf_weight_local,
                hf_target=args.hf_target,
                r2_floor=args.r2_floor,
                r2_weight=r2_weight_local,
                wn_min=args.wn_min,
                wn_max=args.wn_max,
                zeta_min=args.zeta_min,
                zeta_max=args.zeta_max,
            )

        def objective_dyn(x_dyn):
            x_full = base_initial.copy()
            x_full[dyn_indices] = x_dyn
            return compute_global_error(
                x_full,
                all_data,
                hf_weight=hf_weight_local,
                hf_target=args.hf_target,
                r2_floor=args.r2_floor,
                r2_weight=r2_weight_local,
                wn_min=args.wn_min,
                wn_max=args.wn_max,
                zeta_min=args.zeta_min,
                zeta_max=args.zeta_max,
            )

        best_error, best_params = np.inf, base_initial.copy()
        opt_options = {'maxiter': 5000, 'maxfev': 15000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}

        sample_results = []
        for s in range(args.samples):
            if tune_dynamic:
                guess_dyn = base_initial[dyn_indices].copy()
                if s > 0:
                    # wn은 넓게, zeta는 조금 보수적으로 랜덤 시작
                    guess_dyn[0] = base_initial[9] * (0.6 + 1.0 * np.random.rand())
                    guess_dyn[1] = base_initial[10] * (0.6 + 1.0 * np.random.rand())
                    guess_dyn[2] = base_initial[11] * (0.6 + 1.0 * np.random.rand())
                    guess_dyn[3] = base_initial[12] * (0.6 + 1.0 * np.random.rand())
                sample_results.append(np.append(guess_dyn, objective_dyn(guess_dyn)))
            else:
                guess = base_initial.copy() if s == 0 else base_initial * (0.5 + np.random.rand(13))
                if s > 0:
                    guess[4] = (np.random.rand() - 0.5) * 0.1
                    guess[9] = 10 + np.random.rand() * 50
                    guess[11] = 10 + np.random.rand() * 50
                sample_results.append(np.append(guess, objective_full(guess)))

        sample_results = np.array(sample_results)
        top_starts = sample_results[sample_results[:, -1].argsort()][:3, :-1]

        for j, start_val in enumerate(top_starts):
            if tune_dynamic:
                res = minimize(
                    objective_dyn,
                    start_val,
                    method='Powell',
                    bounds=[
                        (args.wn_min, args.wn_max),
                        (args.zeta_min, args.zeta_max),
                        (args.wn_min, args.wn_max),
                        (args.zeta_min, args.zeta_max),
                    ],
                    options={'maxiter': 2000, 'xtol': 1e-3, 'ftol': 1e-3, 'disp': False},
                )
                candidate_params = base_initial.copy()
                candidate_params[dyn_indices] = res.x
            else:
                res = minimize(
                    objective_full,
                    start_val,
                    method='Nelder-Mead',
                    options=opt_options,
                )
                candidate_params = res.x

            if res.fun < best_error:
                best_error = res.fun
                best_params = candidate_params
            print(f"   - 탐색 {j + 1}/3 완료 (오차: {res.fun:.2f})")

        best_params = sanitize_param_vector(
            best_params,
            wn_min=args.wn_min,
            wn_max=args.wn_max,
            zeta_min=args.zeta_min,
            zeta_max=args.zeta_max,
        )
        params_dict = sanitize_params(best_params)

        print(f'\n[결과] {cfg["name"]} 파라미터 도출 완료!')
        print(f"A_max      = {params_dict['a_max']:.6f}")
        print(f"k_shape    = {params_dict['k_shape']:.4f}")
        print(f"C_k        = {params_dict['c_k']:.4f}")
        print(f"C_p        = {params_dict['c_p']:.6f}")
        print(f"C_z        = {params_dict['c_z']:.5f}")
        print(f"A_bw       = {params_dict['a_bw']:.4f}")
        print(f"beta_bw    = {params_dict['beta_bw']:.4f}")
        print(f"gamma_bw   = {params_dict['gamma_bw']:.4f}")
        print(f"alpha_shape= {params_dict['alpha_shape']:.4f}")
        print(f"wn_up      = {params_dict['wn_up']:.4f}")
        print(f"zeta_up    = {params_dict['zeta_up']:.4f}")
        print(f"wn_down    = {params_dict['wn_down']:.4f}")
        print(f"zeta_down  = {params_dict['zeta_down']:.4f}\n")

        optimization_results.append({
            'cfg': cfg,
            'params_dict': params_dict,
            'best_error': best_error,
            'flow_delay_samples': data['flow_delay_samples'],
            'input_window_size': data['input_window_size'],
            'flow_window_size': data['flow_window_size'],
        })

        _, Q_pred = simulate_physics_model(data, best_params)
        q_plot = np.nan_to_num(Q_pred, nan=0.0, posinf=1e9, neginf=-1e9)
        q_plot = np.clip(q_plot, -1e9, 1e9)
        diff_plot = data['Q'] - q_plot
        rmse = np.sqrt(np.mean(diff_plot ** 2))
        ss_tot = np.sum((data['Q'] - np.mean(data['Q'])) ** 2)
        r_sq = 1 - (np.sum(diff_plot ** 2) / ss_tot) if ss_tot != 0 else 0.0
        hf_ratio_fit = high_freq_ratio(data['Q'], Q_pred)
        t_plot = data['Time'] - args.start

        ax = axes[i]

        ax.plot(t_plot, data['Q'], 'k-', linewidth=3, label='Actual Q')
        ax.plot(t_plot, q_plot, 'r--', linewidth=2.2, label='Fitted Q')
        ax.set_yscale('linear')
        title_line1 = (
            f"[{cfg['idx']}] {cfg['name']} | "
            f"Iwin={data['input_window_size']} | Qwin={data['flow_window_size']} | "
            f"delay={data['flow_delay_samples']} samples"
        )
        title_line2 = f"RMSE={rmse:.4f} | R²={r_sq * 100:.1f}% | HF ratio={hf_ratio_fit:.3f}"
        ax.set_title(
            f"{title_line1}\n{title_line2}",
            fontsize=12,
        )
        ax.set_ylabel('Flow', fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True)

    for idx_ax in range(n_plots):
        row_idx = idx_ax // ncols
        if row_idx == nrows - 1:
            axes[idx_ax].set_xlabel('Time [s]', fontsize=12)

    txt_output_path, image_output_path = resolve_result_paths(args.output, args.result_dir)
    output_header = build_output_header(args, selected_valves, optimization_results)
    write_cpp_output(optimization_results, txt_output_path, header_text=output_header)
    fig.savefig(image_output_path, dpi=150)
    print(f"[INFO] C++ make_params 형식 저장 완료: {txt_output_path}")
    print(f"[INFO] fit image 저장 완료: {image_output_path}")

    plt.show()


if __name__ == "__main__":
    main()
