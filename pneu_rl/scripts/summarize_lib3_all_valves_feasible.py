"""Summarize completed type1/type2 feasibility fits without running an optimizer.

Requires all 12 cases and checks the overview CSV against the individual reports,
empirical anchors and static full-open grids before writing summary.md/overview.png.
Rejected fits remain diagnostic candidates and are never relabeled as accepted.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPOSITORY.parent / "pneu_tune/exp/lib3_all_valves_feasible_20261022"
NAMES = ("Supply +", "Supply -", "Act + in", "Act + out", "Act - in", "Act - out")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Unrecognized boolean {value!r}")


def resolve_report(root, row):
    recorded = Path(row["report"])
    candidates = [recorded, REPOSITORY / recorded,
                  root / row["type"] / f"v{int(row['valve'])}" / "report.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No report for {row['type']} valve {row['valve']}")


def load_cases(root):
    rows = read_csv(root / "summary.csv")
    expected = {(kind, valve) for kind in ("type1", "type2") for valve in range(1, 7)}
    keys = [(row["type"], int(row["valve"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        missing = sorted(expected - set(keys))
        raise ValueError(f"Expected exactly 12 distinct completed cases; missing {missing}")
    cases = []
    for row in sorted(rows, key=lambda item: (item["type"], int(item["valve"]))):
        path = resolve_report(root, row)
        report = json.loads(path.read_text())
        label = f"{row['type']} v{row['valve']}"
        if report["type"] != row["type"] or report["valve"] != int(row["valve"]):
            raise ValueError(f"Case identity mismatch: {label}")
        anchors = read_csv(path.with_name("anchors.csv"))
        grid = read_csv(path.with_name("full_open_grid.csv"))
        passed = sum(boolean(anchor["passed"]) for anchor in anchors)
        full_min = min(float(record[column]) for record in grid
                       for column in ("h_minus1", "h_zero", "h_plus1"))
        checks = [
            boolean(row["accepted"]) == report["accepted"],
            passed == int(row["anchors_passed"]) == report["anchors_passed"],
            len(anchors) == int(row["anchors_total"]) == report["anchors_total"],
            np.isclose(full_min, report["full_open_grid_min"], atol=1e-10),
            np.isclose(float(row["rmse"]), report["candidate"]["rmse_lpm"]),
            np.isclose(float(row["r2"]), report["candidate"]["r2"]),
            (full_min >= .95 - 1e-8) == report["full_open_grid_pass"],
        ]
        if not all(checks):
            raise ValueError(f"Summary/report/anchor/grid mismatch: {label}")
        violations = [max(float(anchor["lo"]) - float(anchor["pred_mean"]),
                          float(anchor["pred_mean"]) - float(anchor["hi"]), 0)
                      for anchor in anchors]
        boundary = (not report["accepted"] and report["numerical_valid"]
                    and not report["clipping_used"] and report["full_open_grid_pass"]
                    and max(violations, default=0) <= 1e-5
                    and report["minimum_normalized_margin"] >= -1e-6)
        report.update(_path=path, _anchors=anchors, _full_grid=grid,
                      _boundary_rejection=boundary, _max_anchor_violation=max(violations, default=0))
        cases.append(report)
    return cases


def write_plot(cases, destination):
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    x, width = np.arange(6), .36
    def color(case):
        return "#237e50" if case["accepted"] else ("#cc8c21" if case["_boundary_rejection"] else "#bd493d")
    for column, kind in enumerate(("type1", "type2")):
        selected = [case for case in cases if case["type"] == kind]
        for metric, axis, ylabel in (("rmse_lpm", axes[0, column], "Flow RMSE (L/min)"),
                                     ("r2", axes[1, column], "R squared")):
            axis.bar(x - width / 2, [case["baseline"][metric] for case in selected],
                     width, color=".75", label="Previous fit")
            bars = axis.bar(x + width / 2, [case["candidate"][metric] for case in selected],
                            width, color=[color(case) for case in selected])
            for bar, case in zip(bars, selected):
                if not case["accepted"]:
                    bar.set_hatch("//")
                value = case["candidate"][metric]
                axis.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                              xytext=(0, 4 if value >= 0 else -12),
                              textcoords="offset points", ha="center", fontsize=9)
            axis.set_ylabel(ylabel)
            axis.axhline(0, color=".3", lw=.6)
        axes[0, column].set_title(f"{kind}: {sum(c['accepted'] for c in selected)}/6 accepted")
        axis = axes[2, column]
        axis.bar(x, [case["anchors_passed"] / case["anchors_total"] for case in selected],
                 color=[color(case) for case in selected])
        for index, case in enumerate(selected):
            axis.text(index, .05, f"{case['anchors_passed']}/{case['anchors_total']}",
                      ha="center", fontsize=10)
        axis.set(ylabel="Empirical anchor pass fraction", ylim=(0, 1.08))
        axis.axhline(1, color=".3", ls="--", lw=.7)
        axis = axes[3, column]
        axis.bar(x, [case["full_open_grid_min"] for case in selected],
                 color=["#237e50" if case["full_open_grid_pass"] else "#bd493d"
                        for case in selected])
        axis.axhline(.95, color=".2", ls="--", lw=1)
        for index, case in enumerate(selected):
            axis.text(index, case["full_open_grid_min"] + .025,
                      f"{case['full_open_grid_min']:.6f}", ha="center", fontsize=8)
        axis.set(ylabel="Minimum full-open area ratio (PRIOR)", ylim=(0, 1.15))
        for axis in axes[:, column]:
            axis.set_xticks(x, [f"v{i + 1}\n{NAMES[i]}" for i in x])
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=".75", label="Previous fit"),
                        Patch(facecolor="#237e50", label="Accepted candidate"),
                        Patch(facecolor="#bd493d", hatch="//", label="REJECTED diagnostic candidate"),
                        Patch(facecolor="#cc8c21", hatch="//", label="REJECTED numerical boundary")],
               loc="outside lower center", ncol=2)
    fig.suptitle("Experimental exact-hysteresis fits: separate in-sample calibration", fontsize=15)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def write_markdown(cases, root):
    count = sum(case["accepted"] for case in cases)
    lines = ["# All-valve feasibility calibration results", "",
             f"**Accepted: {count}/12. Rejected: {12 - count}/12.** "
             "Each type was fitted independently on its own recording. These are in-sample results.", "",
             "Rejected candidates are diagnostic outputs and must not be used for pump/RL tuning. "
             "Accepted means the declared calibration gates passed; hardware validity remains "
             "untested. These experimental exact-hysteresis fits require a whole-simulator "
             "update and replay validation before pump/RL use.", "",
             "The 95% full-open area requirement is a model prior, not a measured area. "
             "Empirical flow anchors constrain recorded hold tails and pressure-conditioned "
             "full-open samples. They do not establish closure outside observed conditions.", "",
             "![Overview](overview.png)", ""]
    for kind in ("type1", "type2"):
        selected = [case for case in cases if case["type"] == kind]
        lines.extend([f"## {kind}", "",
                      "| Valve | Status | RMSE old → candidate (L/min) | R² old → candidate | Anchors | Full-open min / prior | Numerical / clipping | Min margin |",
                      "|---|---|---:|---:|---:|---|---|---:|"])
        for case in selected:
            valve, base, candidate = case["valve"], case["baseline"], case["candidate"]
            status = "ACCEPTED" if case["accepted"] else "**REJECTED**"
            if case["_boundary_rejection"]:
                status += " (numerical boundary)"
            full = "PASS" if case["full_open_grid_pass"] else "FAIL"
            numerical = "valid" if case["numerical_valid"] else "INVALID"
            clipping = "USED" if case["clipping_used"] else "none"
            lines.append(f"| [v{valve}: {NAMES[valve - 1]}]({kind}/v{valve}/report.json) | {status} | "
                         f"{base['rmse_lpm']:.3f} → {candidate['rmse_lpm']:.3f} | "
                         f"{base['r2']:.3f} → {candidate['r2']:.3f} | "
                         f"{case['anchors_passed']}/{case['anchors_total']} | "
                         f"{case['full_open_grid_min']:.6f} / {full} | {numerical} / {clipping} | "
                         f"{case['minimum_normalized_margin']:.3g} |")
        lines.extend(["", "Empirical anchor breakdown:", "",
                      "| Valve | Hold tails passed/total | Full-open pressure bins passed/total |",
                      "|---|---:|---:|"])
        for case in selected:
            entries = []
            for group in ("hold_tail", "full_open_pressure_bin"):
                anchors = [anchor for anchor in case["_anchors"] if anchor["kind"] == group]
                entries.append(f"{sum(boolean(a['passed']) for a in anchors)}/{len(anchors)}")
            lines.append(f"| v{case['valve']} | {entries[0]} | {entries[1]} |")
        lines.append("")
        for case in selected:
            if case["_boundary_rejection"]:
                lines.extend([f"v{case['valve']} remains **REJECTED**, but its largest anchor "
                              f"violation is only {case['_max_anchor_violation']:.3g} L/min "
                              f"(minimum normalized margin {case['minimum_normalized_margin']:.3g}); "
                              "the full-open prior passes. This is a numerical boundary result, "
                              "not the larger feasibility mismatch seen in other rejected cases.", ""])
        for case in selected:
            failed = [anchor for anchor in case["_anchors"] if not boolean(anchor["passed"])]
            if not failed:
                continue
            def violation(anchor):
                return max(float(anchor["lo"]) - float(anchor["pred_mean"]),
                           float(anchor["pred_mean"]) - float(anchor["hi"]), 0)
            lines.extend([f"### v{case['valve']} failed empirical anchors", "",
                          f"Showing the largest {min(5, len(failed))} of {len(failed)} mean-envelope violations. "
                          f"[All anchors]({kind}/v{case['valve']}/anchors.csv).", "",
                          "| Kind | Command | Pin / Pout (kPa) | Observed mean | Allowed envelope | Candidate mean (L/min) |",
                          "|---|---:|---:|---:|---:|---:|"])
            for anchor in sorted(failed, key=violation, reverse=True)[:5]:
                lines.append(f"| {anchor['kind']} | {float(anchor['ctrl']):.3f} | "
                             f"{float(anchor['pin']):.1f} / {float(anchor['pout']):.1f} | "
                             f"{float(anchor['real_mean']):.3f} | {float(anchor['lo']):.3f} to "
                             f"{float(anchor['hi']):.3f} | {float(anchor['pred_mean']):.3f} |")
            lines.append("")
    lines.extend(["## Interpretation limits", "",
                  "- Rejected candidates are selected for least maximum constraint violation, "
                  "so their RMSE need not beat the previous fit.",
                  "- Type1/type2 results are separate fits, not cross-validation results.",
                  "- No controller, hardware experiment, held-out trajectory, or full lib3/pump "
                  "replay is performed by this summary script.",
                  "- Dynamic bounds, sensor timing, current scaling and command-history conventions "
                  "remain assumptions documented in each case report.", "",
                  "The numerical-boundary label is diagnostic only: largest anchor violation "
                  "≤1e-5 L/min, minimum normalized margin ≥−1e-6, valid unclipped dynamics, "
                  "and the full-open prior passed. It does not change the stored acceptance flag.", "",
                  "Source summary SHA256: `" + hashlib.sha256((root / "summary.csv").read_bytes()).hexdigest() + "`.", ""])
    (root / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.result_root.resolve()
    cases = load_cases(root)
    write_plot(cases, root / "overview.png")
    write_markdown(cases, root)
    print(f"Summarized all 12 cases: {sum(c['accepted'] for c in cases)} accepted; "
          f"overview.png and summary.md written in {root}")


if __name__ == "__main__":
    main()
