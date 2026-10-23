#!/usr/bin/env python3
"""Summarize completed separate pump fits and optional cross-recording replays.

No simulator or optimizer is run. The initial replay is the first successful
evaluation in each fit's evaluations.jsonl; final metrics come from report.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT.parent / "pneu_tune/exp/lib3_pump_feasible_20261022"
CHANNELS = ("press_pos", "press_neg", "act_pos_press", "act_neg_press",
            *(f"flow{i}" for i in range(1, 7)))
LABELS = ("Ch +", "Ch -", "Act +", "Act -", *(f"Q{i}" for i in range(1, 7)))
KINDS = ("type1", "type2")


def read_history(path: Path) -> list[dict]:
    rows = []
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: incomplete or invalid evaluation record") from exc
        if not row.get("failure") and row.get("datasets") and np.isfinite(row["objective"]):
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no successful initial replay")
    return rows


def dataset_map(record: dict) -> dict[str, dict]:
    datasets = {str(Path(row["path"]).resolve()): row for row in record["datasets"]}
    if len(datasets) != len(record["datasets"]):
        raise ValueError("Repeated dataset paths in a report")
    for dataset in datasets.values():
        if set(dataset["metrics"]) != set(CHANNELS):
            raise ValueError(f"{dataset['path']}: expected four pressure and six flow metrics")
        for channel, values in dataset["metrics"].items():
            for metric in ("mae", "rmse", "bias"):
                if not np.isfinite(values[metric]):
                    raise ValueError(f"{dataset['path']}: nonfinite {channel}/{metric}")
            if values["r2"] is not None and not np.isfinite(values["r2"]):
                raise ValueError(f"{dataset['path']}: nonfinite {channel}/r2")
    return datasets


def load_fits(directory: Path) -> dict[str, dict]:
    required = [directory / kind / name for kind in KINDS
                for name in ("report.json", "evaluations.jsonl")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fit reports are not finished; missing: " + ", ".join(missing))
    result = {}
    domain_path = directory / "whole_model_feasibility.json"
    domains = json.loads(domain_path.read_text()).get("datasets", {}) if domain_path.exists() else {}
    for kind in KINDS:
        case = directory / kind
        report = json.loads((case / "report.json").read_text())
        history = read_history(case / "evaluations.jsonl")
        initial = history[0]
        final_datasets = dataset_map(report)
        initial_datasets = dataset_map(initial)
        if set(initial_datasets) != set(final_datasets):
            raise ValueError(f"{kind}: initial/final dataset paths differ")
        optimizer = report.get("optimizer")
        if optimizer is not None:
            if not np.allclose(optimizer["x"], report["coeff_micro"], rtol=0, atol=1e-12):
                raise ValueError(f"{kind}: selected coefficients differ from optimizer result.x")
            objective = sum(dataset["objective"] for dataset in final_datasets.values())
            if not np.isclose(objective, optimizer["fun"], rtol=1e-8, atol=1e-8):
                raise ValueError(f"{kind}: verification objective differs from optimizer result.fun")
        result[kind] = {"report": report, "history": history, "initial": initial,
                        "initial_datasets": initial_datasets, "final_datasets": final_datasets,
                        "feasibility": domains.get(kind)}
    return result


def metric_rows(fits: dict[str, dict], directory: Path) -> tuple[list[dict], list[dict]]:
    rows, crosses = [], []
    for kind, fit in fits.items():
        for path, final in fit["final_datasets"].items():
            initial = fit["initial_datasets"][path]
            rows.extend(compare_metrics(kind, kind, path, initial, final, "separate_fit"))
    for path in sorted(directory.glob("cross*/report.json")):
        report = json.loads(path.read_text())
        for dataset_path, final in dataset_map(report).items():
            targets = [kind for kind, fit in fits.items() if dataset_path in fit["initial_datasets"]]
            if len(targets) != 1:
                raise ValueError(f"{path}: cross replay target does not match one fit dataset")
            target = targets[0]
            source_matches = [kind for kind, fit in fits.items()
                              if np.allclose(report["coeff_micro"], fit["report"]["coeff_micro"],
                                             rtol=0, atol=1e-10)]
            source = source_matches[0] if len(source_matches) == 1 else "unspecified"
            initial = fits[target]["initial_datasets"][dataset_path]
            rows.extend(compare_metrics(source, target, dataset_path, initial, final,
                                        path.parent.name))
            crosses.append({"directory": path.parent.name, "source": source, "target": target,
                            "objective": final["objective"], "dataset": final,
                            "initial": initial})
    return rows, crosses


def compare_metrics(source, target, path, initial, final, scope):
    for index, channel in enumerate(CHANNELS):
        before, after = initial["metrics"][channel], final["metrics"][channel]
        row = {"scope": scope, "parameter_source": source, "evaluated_type": target,
               "data_path": path, "channel": channel,
               "unit": "kPa" if index < 4 else "L/min"}
        for name in ("mae", "rmse", "r2", "bias"):
            row[f"initial_{name}"] = before[name]
            row[f"final_{name}"] = after[name]
            row[f"change_{name}"] = (after[name] - before[name]) if (
                before[name] is not None and after[name] is not None) else None
        yield row


def overview(fits: dict, destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    width = .36
    for col, kind in enumerate(KINDS):
        fit = fits[kind]
        # Current invocations fit one recording per type. For multiple files,
        # do not average per-file RMSE into an apparently global RMSE.
        if len(fit["final_datasets"]) != 1:
            raise ValueError("Overview requires exactly one recording per type; metrics CSV supports more")
        path, after = next(iter(fit["final_datasets"].items()))
        before = fit["initial_datasets"][path]
        for row, indices in enumerate((range(4), range(4, 10))):
            axis = axes[row, col]
            indices = list(indices)
            x = np.arange(len(indices))
            values_before = [before["metrics"][CHANNELS[j]]["rmse"] for j in indices]
            values_after = [after["metrics"][CHANNELS[j]]["rmse"] for j in indices]
            axis.bar(x - width / 2, values_before, width, color="#b9c1ca", label="Initial pump")
            bars = axis.bar(x + width / 2, values_after, width, color="#2678ad", label="Fitted pump")
            for bar, value in zip(bars, values_after):
                axis.annotate(f"{value:.2f}", (bar.get_x() + width / 2, value),
                              ha="center", xytext=(0, 4), textcoords="offset points", fontsize=9)
            axis.set_xticks(x, [LABELS[j] for j in indices])
            axis.set_ylabel("Pressure RMSE (kPa)" if row == 0 else "Flow RMSE (L/min)")
            domain = fit.get("feasibility")
            domain_label = ("PASS" if domain["positive_absolute_pressure_pass"] else "FAIL") if domain else "not checked"
            axis.set_title(f"{kind}: optimizer {fit['report'].get('fit_status', 'unknown')}\n"
                           f"Absolute pressure domain: {domain_label}")
            axis.grid(axis="y", alpha=.25)
            axis.set_axisbelow(True)
            axis.margins(y=.15)
            axis.legend()
    fig.suptitle("Separate pump calibration: same-recording initial vs fitted replay")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def convergence(fits: dict, destination: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, kind in zip(axes, KINDS):
        fit = fits[kind]
        history = fit["history"]
        x = [record.get("elapsed_sec", index) / 60. for index, record in enumerate(history)]
        y = [record["objective"] for record in history]
        axis.plot(x, y, color=".65", linewidth=.7, marker=".", markersize=2,
                  label="Successful evaluations")
        axis.plot(x, np.minimum.accumulate(y), color="#2678ad", linewidth=1.8,
                  label="Best objective so far")
        axis.set_title(f"{kind}: {len(history)} successful evaluations, "
                       f"status={fit['report'].get('fit_status', 'unknown')}")
        axis.set_ylabel("Weighted MAE objective")
        axis.set_xlabel("Elapsed time (min)")
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(fits: dict, crosses: list, destination: Path) -> None:
    lines = ["# Lib3 pump calibration", "",
             "Type1 and type2 were fitted separately on their own recordings. "
             "The comparisons below are in-sample initial-versus-fitted replays, "
             "not held-out validation or real controller results.", "",
             "Valve parameters are fixed during pump tuning. The objective retains "
             "1.5×MAE(chamber +) + MAE(chamber −) + 0.1×MAE(Q1) + 0.1×MAE(Q2). "
             "All four pressures and six flows are reported separately; improvements "
             "in the objective do not imply every actuator/flow channel improved.", "",
             "Pressure and flow use a common physical replay clock with no observation "
             "delay or fitted offsets. Commands are recorded ctrl1..6 with a left hold. "
             "Repeated timestamps keep the last recorded row; source CSVs are unchanged.", "",
             "| Fit | Initial inlet / outlet ×10⁻⁶ | Fitted inlet / outlet ×10⁻⁶ | Objective initial → fitted | Improvement | nit / nfev | Status | Elapsed (min) |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    domain_lines = ["**Optimizer convergence is separate from physical feasibility.**", "",
                    "| Fit | Positive absolute pressure | Minimum Act+ (kPa absolute) | Act+ below zero (s) |",
                    "|---|---|---:|---:|"]
    for kind in KINDS:
        domain = fits[kind].get("feasibility")
        if domain:
            pressure = domain["pressure_domain"]["act_pos_press"]
            domain_lines.append(f"| {kind} | {'PASS' if domain['positive_absolute_pressure_pass'] else 'FAIL'} | "
                                f"{pressure['min_kpa']:.3f} | {pressure['negative_duration_sec']:.2f} |")
    if len(domain_lines) > 4:
        domain_lines.extend(["", "[Whole-model feasibility audit](whole_model_feasibility.json). "
                             "Negative absolute pressure is a model failure. These fits are experimental "
                             "results, not adopted RL environments.", ""])
        lines[2:2] = domain_lines
    corrected = destination.parent.parent / "lib3_physical_model_20261023/summary.md"
    if corrected.exists():
        lines[2:2] = ["후속 수정: [개구율 동특성 모델과 물리 검증](../lib3_physical_model_20261023/summary.md). "
                      "아래 FAIL 판정은 수정 전 유량 동특성 모델의 기록이다.", ""]
    for kind in KINDS:
        fit = fits[kind]
        report, initial = fit["report"], fit["initial"]
        optimizer = report.get("optimizer") or {}
        final_objective = sum(row["objective"] for row in report["datasets"])
        reduction = 100 * (1 - final_objective / initial["objective"]) if initial["objective"] else None
        coeff_before = " / ".join(f"{v:.6f}" for v in initial["coeff_micro"])
        coeff_after = " / ".join(f"{v:.6f}" for v in report["coeff_micro"])
        lines.append(f"| [{kind}]({kind}/report.json) | {coeff_before} | {coeff_after} | "
                     f"{initial['objective']:.4f} → {final_objective:.4f} | {number(reduction, 1)}% | "
                     f"{optimizer.get('nit', 'n/a')} / {optimizer.get('nfev', 'n/a')} | "
                     f"{report.get('fit_status', 'unknown')} | {report['elapsed_sec'] / 60:.2f} |")
    lines.extend(["", "![Initial and fitted channel RMSE](overview.png)", "",
                  "![Optimizer convergence](convergence.png)", "",
                  "[All initial/final metrics and changes (CSV)](before_after_metrics.csv)", ""])
    for kind in KINDS:
        fit = fits[kind]
        report = fit["report"]
        optimizer = report.get("optimizer") or {}
        lines.extend([f"## {kind}", "", f"Optimizer: {optimizer.get('message', 'evaluation only')}. "
                      f"Native library hash unchanged: {report.get('library_hash_unchanged', 'not recorded')}.", ""])
        for path, final in fit["final_datasets"].items():
            initial = fit["initial_datasets"][path]
            lines.extend([f"Recording: `{Path(path).name}`", "",
                          "| Channel | Unit | RMSE initial → fitted | MAE initial → fitted | R² initial → fitted |",
                          "|---|---|---:|---:|---:|"])
            for index, channel in enumerate(CHANNELS):
                before, after = initial["metrics"][channel], final["metrics"][channel]
                comparisons = [f"{number(before[m])} → {number(after[m])}" for m in ("rmse", "mae", "r2")]
                lines.append(f"| {channel} | {'kPa' if index < 4 else 'L/min'} | " + " | ".join(comparisons) + " |")
            lines.append("")
    if crosses:
        lines.extend(["## Optional cross-recording replay", "",
                      "Each recording was used to fit its own model earlier. Applying one fit "
                      "to the other recording checks parameter sensitivity; it does not turn "
                      "either recording into held-out data. The entire fixed library may differ "
                      "between types, so this is not an isolated pump-coefficient comparison.", "",
                      "| Replay | Parameter source → recording | Objective | Ch+ RMSE | Ch− RMSE | Act+ RMSE | Act− RMSE |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for cross in crosses:
            pressures = [cross["dataset"]["metrics"][channel]["rmse"] for channel in CHANNELS[:4]]
            lines.append(f"| [{cross['directory']}]({cross['directory']}/report.json) | "
                         f"{cross['source']} → {cross['target']} | {cross['objective']:.4f} | "
                         + " | ".join(f"{v:.3f}" for v in pressures) + " |")
        lines.append("")
    lines.extend(["A run with status `incomplete` reached its stopping limit without declared "
                  "convergence. Its best selected coefficients are still reported, without "
                  "labeling the fit converged. R² can be very negative on near-constant "
                  "channels; compare absolute RMSE/MAE alongside R².", ""])
    destination.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    directory = args.root.resolve()
    try:
        fits = load_fits(directory)
        rows, crosses = metric_rows(fits, directory)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Summary not written: {exc}\n")
    with (directory / "before_after_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overview(fits, directory / "overview.png")
    convergence(fits, directory / "convergence.png")
    markdown(fits, crosses, directory / "summary.md")
    print(f"Wrote {directory / 'summary.md'} ({len(rows)} channel comparisons)")


if __name__ == "__main__":
    main()
