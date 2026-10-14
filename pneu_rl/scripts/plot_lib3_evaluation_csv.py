"""Regenerate standard lib3 main/control plots from saved evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from pneu_rl.plotting import plot_lib3_pressure_and_controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()

    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        prefix = csv_path.with_suffix("")
        main_path, ctrl_path = plot_lib3_pressure_and_controls(df, prefix)
        print(f"saved: {main_path}")
        print(f"saved: {ctrl_path}")


if __name__ == "__main__":
    main()
