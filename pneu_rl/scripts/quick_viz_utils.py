from typing import Mapping

import numpy as np
import pandas as pd


def datas_to_df(datas: Mapping) -> pd.DataFrame:
    return pd.DataFrame({key: list(value) for key, value in datas.items()})


def _tracking_columns(df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    if {"ref_act_pos", "ref_act_neg", "act_pos_press", "act_neg_press"}.issubset(df.columns):
        return (
            "ref_act_pos",
            "ref_act_neg",
            "act_pos_press",
            "act_neg_press",
            "Actuator",
        )
    if {"ref_pos", "ref_neg", "sen_pos", "sen_neg"}.issubset(df.columns):
        return (
            "ref_pos",
            "ref_neg",
            "sen_pos",
            "sen_neg",
            "Chamber",
        )
    raise KeyError(
        "tracking columns not found. Expected either "
        "ref_pos/ref_neg/sen_pos/sen_neg or "
        "ref_act_pos/ref_act_neg/act_pos_press/act_neg_press."
    )


def _err_metrics(pos_err: np.ndarray, neg_err: np.ndarray) -> dict:
    return dict(
        rmse_pos=float(np.sqrt(np.mean(pos_err * pos_err))),
        rmse_neg=float(np.sqrt(np.mean(neg_err * neg_err))),
        mae_pos=float(np.mean(np.abs(pos_err))),
        mae_neg=float(np.mean(np.abs(neg_err))),
        bias_pos=float(np.mean(pos_err)),
        bias_neg=float(np.mean(neg_err)),
    )


def compute_tracking_metrics(
    datas: Mapping,
    window_start: float = 30.0,
    window_end: float = 90.0,
) -> dict:
    df = datas_to_df(datas)
    if len(df) == 0:
        return dict(
            tracking_pair="unknown",
            sample_count=0,
            full=dict(),
            window=dict(
                start_sec=window_start,
                end_sec=window_end,
                sample_count=0,
            ),
        )

    ref_pos_col, ref_neg_col, obs_pos_col, obs_neg_col, pair_name = _tracking_columns(df)

    ref_pos = np.asarray(df[ref_pos_col], dtype=np.float64)
    ref_neg = np.asarray(df[ref_neg_col], dtype=np.float64)
    obs_pos = np.asarray(df[obs_pos_col], dtype=np.float64)
    obs_neg = np.asarray(df[obs_neg_col], dtype=np.float64)

    pos_err = ref_pos - obs_pos
    neg_err = ref_neg - obs_neg

    metrics = dict(
        tracking_pair=pair_name,
        sample_count=int(len(df)),
        columns=dict(
            ref_pos=ref_pos_col,
            ref_neg=ref_neg_col,
            obs_pos=obs_pos_col,
            obs_neg=obs_neg_col,
        ),
        full=_err_metrics(pos_err, neg_err),
        window=dict(
            start_sec=float(window_start),
            end_sec=float(window_end),
        ),
    )

    if "curr_time" in df.columns:
        time_arr = np.asarray(df["curr_time"], dtype=np.float64)
        idx = np.where((time_arr >= window_start) & (time_arr <= window_end))[0]
    else:
        idx = np.array([], dtype=np.int64)

    metrics["window"]["sample_count"] = int(len(idx))
    if len(idx) > 0:
        metrics["window"].update(_err_metrics(pos_err[idx], neg_err[idx]))

    return metrics


def print_tracking_metrics(metrics: Mapping) -> None:
    print("[ INFO] Tracking RMSE")
    print(f"  Pair: {metrics.get('tracking_pair', 'unknown')}")

    full = metrics.get("full", {})
    if full:
        print(
            "  Full: "
            f"pos {full['rmse_pos']:.4f} kPa, "
            f"neg {full['rmse_neg']:.4f} kPa "
            f"(MAE pos {full['mae_pos']:.4f}, neg {full['mae_neg']:.4f})"
        )

    window = metrics.get("window", {})
    if window.get("sample_count", 0) > 0:
        print(
            f"  {window['start_sec']:.0f}s ~ {window['end_sec']:.0f}s: "
            f"pos {window['rmse_pos']:.4f} kPa, "
            f"neg {window['rmse_neg']:.4f} kPa "
            f"(n={window['sample_count']})"
        )
    else:
        print(
            f"  {window.get('start_sec', 30.0):.0f}s ~ "
            f"{window.get('end_sec', 90.0):.0f}s: no samples"
        )
