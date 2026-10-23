"""Compare frozen-coefficient flow-lag and opening-lag lib3 replays."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("press_pos", "press_neg", "act_pos_press", "act_neg_press",
            *(f"flow{i}" for i in range(1, 7)))
LABELS = ("Ch+", "Ch-", "Act+", "Act-", *(f"Q{i}" for i in range(1, 7)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT.parent/"pneu_tune/exp/lib3_physical_model_20261023")
    parser.add_argument("--baseline", type=Path, default=ROOT.parent/"pneu_tune/exp/lib3_pump_feasible_20261022")
    args = parser.parse_args()
    validation = json.loads((args.root/"validation.json").read_text())
    metrics, cases = [], {}
    for kind in ("type1", "type2"):
        before = json.loads((args.baseline/kind/"report.json").read_text())
        after = json.loads((args.root/"replay"/kind/"report.json").read_text())
        if before["coeff_micro"] != after["coeff_micro"]:
            raise ValueError(f"{kind}: pump parameters changed")
        old = before["datasets"][0]["metrics"]
        new = after["datasets"][0]["metrics"]
        old_trace = pd.read_csv(args.baseline/kind/"replay_1.csv")
        new_trace = pd.read_csv(args.root/"replay"/kind/"replay_1.csv")
        np.testing.assert_allclose(old_trace.time, new_trace.time, rtol=0, atol=1e-12)
        for name in CHANNELS:
            np.testing.assert_allclose(old_trace[f"real_{name}"], new_trace[f"real_{name}"], atol=1e-10)
            metrics.append(dict(type=kind, channel=name, unit="kPa" if name in CHANNELS[:4] else "L/min",
                                old_rmse=old[name]["rmse"], new_rmse=new[name]["rmse"],
                                old_r2=old[name]["r2"], new_r2=new[name]["r2"],
                                old_mae=old[name]["mae"], new_mae=new[name]["mae"]))
        steps = validation["kinds"][kind]["steps"]
        passed = all(
            step["domain"]["finite"] and step["domain"]["strictly_positive_absolute_pressure"]
            and step["domain"]["nonnegative_valve_flow"] and step["domain"]["wrong_head_nonzero_flow_samples"] == 0
            and step["state_domain"]["all_six_gas_pressures_positive"]
            and step["state_domain"]["h_bounded"] and step["state_domain"]["opening_bounded"]
            and all(s["strictly_positive_absolute_pressure"] and s["wrong_head_nonzero_flow_samples"] == 0
                    and s["state_domain"]["all_six_gas_pressures_positive"]
                    and s["state_domain"]["opening_bounded"] and s["state_domain"]["h_bounded"]
                    for s in step["synthetic"].values()) for step in steps)
        cases[kind] = dict(old=old, new=new, old_trace=old_trace, new_trace=new_trace,
                           tested_feasibility_pass=passed,
                           minimum_pressure_before=float(old_trace.sim_act_pos_press.min()),
                           minimum_pressure_after=float(new_trace.sim_act_pos_press.min()))

    pd.DataFrame(metrics).to_csv(args.root/"comparison_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, kind in enumerate(cases):
        case = cases[kind]
        for row, indices in enumerate((range(4), range(4, 10))):
            ix = list(indices)
            x, width = np.arange(len(ix)), .36
            axes[row, col].bar(x-width/2, [case["old"][CHANNELS[i]]["rmse"] for i in ix], width,
                              label="Flow dynamics", color="#c5a38f")
            bars = axes[row, col].bar(x+width/2, [case["new"][CHANNELS[i]]["rmse"] for i in ix], width,
                                     label="Opening dynamics", color="#247aab")
            for bar in bars:
                axes[row, col].annotate(f"{bar.get_height():.2f}",
                    (bar.get_x()+width/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center")
            axes[row, col].set_xticks(x, [LABELS[i] for i in ix])
            axes[row, col].set_ylabel("RMSE (kPa)" if row == 0 else "RMSE (L/min)")
            axes[row, col].set_title(kind+": frozen parameters, changed state dynamics")
            axes[row, col].grid(axis="y", alpha=.2)
            axes[row, col].set_axisbelow(True)
            axes[row, col].margins(y=.2)
            axes[row, col].legend()
    fig.savefig(args.root/"comparison_rmse.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for col, kind in enumerate(cases):
        case = cases[kind]
        old, new = case["old_trace"], case["new_trace"]
        for row in (0, 1):
            axis = axes[row, col]
            axis.plot(new.source_time, new.real_act_pos_press, color="black", lw=.8, label="Real")
            axis.plot(old.source_time, old.sim_act_pos_press, color="#c8825d", lw=.8, alpha=.7, label="Flow lag")
            axis.plot(new.source_time, new.sim_act_pos_press, color="#247aab", lw=.9, label="Opening lag")
            axis.axhline(0., color="red", ls=":", lw=.6)
            axis.set_ylabel("Act+ absolute pressure (kPa)")
            axis.set_title(kind+ (": full recording" if row == 0 else ": same commanded ramp"))
            axis.grid(alpha=.2)
            axis.legend()
            if row == 1:
                axis.set_xlim((99, 130) if kind == "type1" else (186, 196))
            axis.set_xlabel("Source time (s)")
    fig.savefig(args.root/"comparison_pressure.png", dpi=150)
    plt.close(fig)

    lines = ["# Lib3: physical opening dynamics", "",
             "기억하는 상태를 유량에서 개구율로 옮기고, 현재 압력에서 유량을 계산했다. "
             "RK 중간 계산에서 상태를 갱신하지 않도록 분리했다. 압력 clipping·새 누설항·센서 보정은 없다.", "",
             "밸브 계수·펌프 계수는 고정했다. 시간상수는 기존 계수에서 τ=2ζ/ωn으로 변환한 "
             "초기값이며 새 구조로 재튜닝한 결과가 아니다. 같은 기록에서의 비교이며 실물 제어 검증도 아니다.", "",
             "| Type | 이전 act+ 최소 절대압 | 수정 후 | 검사 범위 feasibility |",
             "|---|---:|---:|---|"]
    for kind, case in cases.items():
        lines.append(f"| {kind} | {case['minimum_pressure_before']:.3f}kPa | "
                     f"{case['minimum_pressure_after']:.6f}kPa | {'통과' if case['tested_feasibility_pass'] else '미통과'} |")
    lines += ["", "[검증 상세](validation.json): 두 전체 기록 × 내부 간격0.1/0.05/0.025ms, "
              "각 간격의5개 상수 명령 시험에서6개 기체 압력 양수·개구율 범위·역차압 유량을 확인했다. "
              "임의의 모든 상태/입력에 대한 전역 보장은 아니다.", "",
              "**0.8 명령은 완전 폐쇄나 압력 유지로 가정하지 않는다.** "
              "상수 저명령에서도 기존 정적 밸브식에 따른 흐름이 남을 수 있고, 펌프 ON·유출이 작은 경우 공급 압력이 올라간다.", "",
              "![동일 계수 정확도 비교](comparison_rmse.png)", "",
              "![압력 시계열 비교](comparison_pressure.png)", "",
              "| Type | Channel | RMSE 이전→수정 | R² 이전→수정 |", "|---|---|---:|---:|"]
    for item in metrics:
        lines.append(f"| {item['type']} | {item['channel']} ({item['unit']}) | "
                     f"{item['old_rmse']:.3f}→{item['new_rmse']:.3f} | {item['old_r2']:.3f}→{item['new_r2']:.3f} |")
    lines += ["", "[전체 수치 CSV](comparison_metrics.csv)", "",
              "- [기존 진동 분석](oscillation_baseline_audit.json)",
              "- [수정 후 같은 구간 분석](oscillation_opening_audit.json)", "",
              "이전 snapshot의 유량 상태와 새 개구율 상태는 호환되지 않는다. "
              "새 라이브러리는 state schema2와 checked restore API를 사용한다."]
    (args.root/"summary.md").write_text("\n".join(lines)+"\n")
    status = dict(cases={kind: {key: case[key] for key in (
        "tested_feasibility_pass", "minimum_pressure_before", "minimum_pressure_after")}
        for kind, case in cases.items()}, parameters_refitted=False,
        actual_hardware_validated=False, actuator_volumes_l=dict(positive=.1, negative=.156),
        note="Feasibility in the inspected cases; new time constants mapped from old flow dynamics.")
    (args.root/"status.json").write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
