"""Shared publication-style plotting helpers for lib3 pressure control."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
_FONT = fm.FontProperties(fname=str(_FONT_PATH)) if _FONT_PATH.is_file() else None
_LABEL_FONT_SIZE = 18
_PRESSURE_LINE_WIDTH = 2.0
_CONTROL_LINE_WIDTH = 1.8


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"none of the required columns are present: {tuple(names)}")


def _style_axis(axis, ylabel: str) -> None:
    axis.set_xlabel("Time [sec]", fontproperties=_FONT, fontsize=_LABEL_FONT_SIZE)
    axis.set_ylabel(ylabel, fontproperties=_FONT, fontsize=_LABEL_FONT_SIZE)
    axis.grid(which="major", color="silver", linewidth=1.0)
    axis.grid(which="minor", color="lightgray", linewidth=0.5)
    axis.minorticks_on()
    axis.tick_params(axis="x", which="both", labelbottom=True)
    axis.set_xlim(left=0.0)


def plot_lib3_pressure_and_controls(
    df: pd.DataFrame,
    output_prefix: Path | None,
    *,
    show: bool = False,
) -> tuple[Path | None, Path | None]:
    """Save the standard lib3 main and six-channel control figures.

    The main figure always contains exactly three pressure panels: both chamber
    pressures without references, actuator positive with its reference, and
    actuator negative with its reference.  Lib3 has no displacement panel.
    Each physical valve command is placed in its own control subplot.
    """

    time_col = _first_present(df, ("time", "curr_time"))
    chamber_pos_col = _first_present(df, ("chamber_pos", "press_pos"))
    chamber_neg_col = _first_present(df, ("chamber_neg", "press_neg"))
    act_pos_col = _first_present(df, ("act_pos", "act_pos_press"))
    act_neg_col = _first_present(df, ("act_neg", "act_neg_press"))
    ref_pos_col = _first_present(df, ("ref_pos", "ref_act_pos"))
    ref_neg_col = _first_present(df, ("ref_neg", "ref_act_neg"))
    control_columns = [
        _first_present(df, ("ctrl1", "ctrl_ch_pos")),
        _first_present(df, ("ctrl2", "ctrl_ch_neg")),
        _first_present(df, ("ctrl3", "ctrl_act_pos_in")),
        _first_present(df, ("ctrl4", "ctrl_act_pos_out")),
        _first_present(df, ("ctrl5", "ctrl_act_neg_in")),
        _first_present(df, ("ctrl6", "ctrl_act_neg_out")),
    ]

    time = df[time_col].to_numpy(dtype=np.float64)
    main_path = None
    ctrl_path = None
    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        main_path = output_prefix.with_name(f"{output_prefix.name}_main.png")
        ctrl_path = output_prefix.with_name(f"{output_prefix.name}_ctrl.png")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(
        time,
        df[chamber_pos_col],
        linewidth=_PRESSURE_LINE_WIDTH,
        color="red",
        label="CH_POS",
    )
    axes[0].plot(
        time,
        df[chamber_neg_col],
        linewidth=_PRESSURE_LINE_WIDTH,
        color="blue",
        label="CH_NEG",
    )
    _style_axis(axes[0], "Chamber [kPa]")

    axes[1].plot(
        time,
        df[ref_pos_col],
        linewidth=_PRESSURE_LINE_WIDTH,
        color="black",
        label="REF",
    )
    axes[1].plot(
        time,
        df[act_pos_col],
        linewidth=_PRESSURE_LINE_WIDTH,
        color="red",
        label="ACT_POS",
    )
    _style_axis(axes[1], "Act Pos [kPa]")

    axes[2].plot(
        time,
        df[ref_neg_col],
        linewidth=_PRESSURE_LINE_WIDTH,
        color="black",
        label="REF",
    )
    axes[2].plot(
        time,
        df[act_neg_col],
        linewidth=_PRESSURE_LINE_WIDTH,
        color="blue",
        label="ACT_NEG",
    )
    _style_axis(axes[2], "Act Neg [kPa]")

    for axis in axes:
        axis.legend(loc="upper right")
    fig.tight_layout()
    if main_path is not None:
        fig.savefig(main_path, dpi=150)

    control_style = [
        ("CH POS\nControl [-]", "#d62728"),
        ("CH NEG\nControl [-]", "#1f77b4"),
        ("ACT POS IN\nControl [-]", "#2ca02c"),
        ("ACT POS OUT\nControl [-]", "#ff7f0e"),
        ("ACT NEG IN\nControl [-]", "#9467bd"),
        ("ACT NEG OUT\nControl [-]", "#8c564b"),
    ]
    ctrl_fig, ctrl_axes = plt.subplots(6, 1, figsize=(9, 14), sharex=True)
    for axis, column, (ylabel, color) in zip(
        ctrl_axes, control_columns, control_style
    ):
        axis.plot(
            time,
            df[column],
            linewidth=_CONTROL_LINE_WIDTH,
            color=color,
        )
        _style_axis(axis, ylabel)
    ctrl_fig.tight_layout()
    if ctrl_path is not None:
        ctrl_fig.savefig(ctrl_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(ctrl_fig)
    return main_path, ctrl_path
